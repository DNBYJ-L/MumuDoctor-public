#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
家庭组：账号之间的关联请求与数据授权。

一个账号可以邀请另一个账号组成「家庭组」：
    主账号 owner 发起请求 → 被邀请账号 member 接受 / 拒绝
    接受后 owner 才能查看 member 的打卡、看板和问询历史，并可代 member 打卡。

一个家庭组只有一个管理者（发起邀请的 owner），但可以邀请多位家人；
一个账号也只能属于一个家庭组：要么作为管理者，要么作为成员。
（已有的历史数据不做删除，继续按 active 关系显示；新的邀请与接受会按这条规则校验。）

family_links 一张表存所有关系（没有外键，和 users/checkins 一样由应用层校验）：
    owner_id / member_id   主账号 / 被关联账号
    relation               申请人相对被申请人的直系亲属关系：
                           child 我是对方的子女 / parent 我是对方的父母 / spouse 配偶
    status                 pending 等待确认 / active 已授权 / rejected 已拒绝
    message                邀请留言（可选）
    created_at / responded_at / updated_at

称谓按双方注册时填写的性别自动换算：
    申请人选「子女」→ 申请人看到对方是「父亲 / 母亲」，
    对方看到申请人是「儿子 / 女儿」。

权限只认 status='active'；pending 和 rejected 一律不能读数据。
同一个 owner 对同一个 member 只有一行，拒绝后可以重新邀请（回到 pending）。
"""

from __future__ import annotations

import logging
from contextlib import contextmanager
from typing import Dict, Iterator, List, Optional

import auth

logger = logging.getLogger(__name__)

STATUS_PENDING = "pending"
STATUS_ACTIVE = "active"
STATUS_REJECTED = "rejected"

# 申请人相对被申请人的关系。只开放直系亲属，避免出现「表舅」「同事」这类
# 无法自动换算称谓、也难以校验的关系。
RELATION_CHILD = "child"
RELATION_PARENT = "parent"
RELATION_SPOUSE = "spouse"
RELATIONS = (RELATION_CHILD, RELATION_PARENT, RELATION_SPOUSE)

RELATION_LABELS = {
    RELATION_CHILD: "子女",
    RELATION_PARENT: "父母",
    RELATION_SPOUSE: "配偶",
}
RELATION_OPTION_LABELS = {
    RELATION_CHILD: "子女（我是对方的子女）",
    RELATION_PARENT: "父母（我是对方的父母）",
    RELATION_SPOUSE: "配偶（我是对方的配偶）",
}
# 反向关系：申请人选「子女」，对方相对申请人就是「父母」
_REVERSE_RELATION = {
    RELATION_CHILD: RELATION_PARENT,
    RELATION_PARENT: RELATION_CHILD,
    RELATION_SPOUSE: RELATION_SPOUSE,
}
_TITLES = {
    RELATION_CHILD: {"male": "儿子", "female": "女儿", "other": "子女"},
    RELATION_PARENT: {"male": "父亲", "female": "母亲", "other": "父母"},
    RELATION_SPOUSE: {"male": "丈夫", "female": "妻子", "other": "配偶"},
}
FALLBACK_TITLE = "家人"

MESSAGE_MAX = 120

_SCHEMA = """
CREATE TABLE IF NOT EXISTS family_links (
    id BIGINT UNSIGNED NOT NULL AUTO_INCREMENT,
    owner_id BIGINT UNSIGNED NOT NULL,
    member_id BIGINT UNSIGNED NOT NULL,
    status ENUM('pending', 'active', 'rejected') NOT NULL DEFAULT 'pending',
    message VARCHAR(120) NOT NULL DEFAULT '',
    relation VARCHAR(16) NOT NULL DEFAULT '',
    created_at DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP,
    responded_at DATETIME NULL,
    updated_at DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP,
    PRIMARY KEY (id),
    UNIQUE KEY uk_family_pair (owner_id, member_id),
    KEY idx_family_member_status (member_id, status),
    KEY idx_family_owner_status (owner_id, status)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci
"""

_schema_ready = False


class FamilyError(Exception):
    """可以直接展示给用户的家庭组关怀错误。"""


class ValidationError(FamilyError):
    """邀请 / 响应 / 解除时的不合法操作。"""


class DatabaseError(FamilyError):
    """数据库连不上或读写失败；细节只写服务端日志。"""


def _ensure_schema(cur) -> None:
    global _schema_ready
    if _schema_ready:
        return
    try:
        cur.execute(_SCHEMA)
        # 老库的 family_links 没有 relation 列：CREATE TABLE IF NOT EXISTS 不会补列，
        # 这里显式检查一次并 ALTER，升级时不需要手工改表。
        cur.execute(
            "SELECT COUNT(*) AS n FROM information_schema.columns"
            " WHERE table_schema = DATABASE() AND table_name = 'family_links'"
            " AND column_name = 'relation'"
        )
        row = cur.fetchone() or {}
        if not int(row.get("n") or 0):
            cur.execute(
                "ALTER TABLE family_links ADD COLUMN relation VARCHAR(16)"
                " NOT NULL DEFAULT '' AFTER message"
            )
        _schema_ready = True
    except Exception as exc:  # noqa: BLE001
        logger.warning("ensure family schema failed: %s", exc)
        raise DatabaseError("家庭组数据表初始化失败，请检查数据库权限。") from exc


@contextmanager
def _cursor() -> Iterator:
    """统一把底层 auth.DatabaseError 翻译成 FamilyError。"""
    try:
        with auth.db_cursor() as cur:
            _ensure_schema(cur)
            yield cur
    except auth.DatabaseError as exc:
        raise DatabaseError(str(exc)) from exc


def _normalize_account(account: str) -> str:
    account = auth.normalize_account(account)
    if not account:
        raise ValidationError("请填写对方的账号。")
    if len(account) > auth.ACCOUNT_MAX:
        raise ValidationError(f"账号最多 {auth.ACCOUNT_MAX} 个字符。")
    return account


def _clean_message(message: str) -> str:
    return (message or "").strip()[:MESSAGE_MAX]


def normalize_relation(relation: str) -> str:
    """校验邀请时选择的关系；空值不能通过。"""
    value = (relation or "").strip()
    if value not in RELATIONS:
        raise ValidationError("请选择你与对方的关系（只支持父母、子女、配偶）。")
    return value


def relation_title(relation: str, gender: str, reverse: bool = False) -> str:
    """把关系换算成「父亲 / 女儿 / 妻子」这类称谓。

    reverse=False：看「本人」——申请人选 child，本人是对方的儿子/女儿；
    reverse=True ：看「对方」——申请人选 child，对方是本人的父亲/母亲。
    性别取注册时填写的那一项，未填写/其他时退回中性称谓。
    """
    key = str(relation or "")
    if reverse:
        key = _REVERSE_RELATION.get(key, key)
    return _TITLES.get(key, {}).get(str(gender or ""), FALLBACK_TITLE)


def _row_to_link(row: Dict, perspective: str = "owner") -> Dict:
    """统一成前端好用的结构：id/user_id 是「对方」的账号 id，link_id 是关系行 id。

    perspective="owner" 表示当前查询是主账号在看成员（称谓要反向换算）；
    perspective="member" 表示成员在看主账号（称谓按申请人填的关系直接显示）。
    """
    counterpart = row.get("user_id")
    if counterpart is None:
        # 兼容只查了 member_id 的旧查询
        counterpart = row.get("member_id")
    gender = str(row.get("gender") or "")
    relation = str(row.get("relation") or "")
    return {
        "id": int(counterpart),
        "user_id": int(counterpart),
        "link_id": int(row["id"]),
        "owner_id": int(row["owner_id"]),
        "member_id": int(row["member_id"]),
        "status": str(row["status"]),
        "relation": relation,
        "relation_label": RELATION_LABELS.get(relation, FALLBACK_TITLE),
        "relation_title": relation_title(
            relation, gender, reverse=(perspective == "owner")
        ),
        "message": str(row.get("message") or ""),
        "created_at": row.get("created_at"),
        "responded_at": row.get("responded_at"),
        "account": str(row.get("account") or ""),
        "gender": gender,
        "age": int(row["age"]) if row.get("age") is not None else 0,
        "linked_at": row.get("linked_at") or row.get("responded_at"),
    }


def _active_group_link(cur, user_id: int) -> Optional[Dict]:
    """这个账号已经加入的家庭组（任何 active 关系都算）。"""
    cur.execute(
        "SELECT id, owner_id, member_id FROM family_links"
        " WHERE status = %s AND (owner_id = %s OR member_id = %s) LIMIT 1",
        (STATUS_ACTIVE, int(user_id), int(user_id)),
    )
    return cur.fetchone()


def _active_membership(cur, user_id: int) -> Optional[Dict]:
    """这个账号是否已经是某个家庭组的成员（不是管理者）。"""
    cur.execute(
        "SELECT id, owner_id FROM family_links"
        " WHERE status = %s AND member_id = %s LIMIT 1",
        (STATUS_ACTIVE, int(user_id)),
    )
    return cur.fetchone()


def _find_user(cur, account: str) -> Optional[Dict]:
    cur.execute(
        "SELECT id, account, gender, age FROM users WHERE account = %s",
        (account,),
    )
    return cur.fetchone()


def invite(owner_id: int, account: str, relation: str, message: str = "") -> Dict:
    """主账号按账号名发起关联请求；重复邀请 / 自关联都会给出明确提示。

    relation 是申请人相对被申请人的直系亲属关系，发起时必须选择。
    """
    account = _normalize_account(account)
    relation = normalize_relation(relation)
    message = _clean_message(message)
    owner_id = int(owner_id)

    try:
        with _cursor() as cur:
            target = _find_user(cur, account)
            if not target:
                raise ValidationError("没有找到这个账号，请确认对方已经注册。")
            member_id = int(target["id"])
            if member_id == owner_id:
                raise ValidationError("不能关联自己的账号。")

            cur.execute(
                "SELECT * FROM family_links WHERE owner_id = %s AND member_id = %s",
                (owner_id, member_id),
            )
            existing = cur.fetchone()
            if existing:
                status = str(existing["status"])
                if status == STATUS_ACTIVE:
                    raise ValidationError("你们已经建立了家庭组关联，不需要重复邀请。")
                if status == STATUS_PENDING:
                    raise ValidationError("已经发过请求了，正在等对方确认。")

            # 一个账号只能属于一个家庭组：自己不能已经是别组的成员；
            # 对方也不能已经在任何家庭组里（否则接受后会出现一人两组）。
            owner_membership = _active_membership(cur, owner_id)
            if owner_membership:
                raise ValidationError(
                    "你已经加入了另一个家庭组，不能再作为管理者创建家庭组。"
                )
            target_group = _active_group_link(cur, member_id)
            if target_group:
                raise ValidationError("对方已经加入了一个家庭组，暂时不能关联。")

            if existing:
                # 之前被拒绝：允许重新发起，回到 pending，并更新这次选的关系
                cur.execute(
                    "UPDATE family_links SET status = %s, relation = %s, message = %s,"
                    " created_at = NOW(), responded_at = NULL WHERE id = %s",
                    (STATUS_PENDING, relation, message, int(existing["id"])),
                )
                link_id = int(existing["id"])
            else:
                cur.execute(
                    "INSERT INTO family_links"
                    " (owner_id, member_id, status, relation, message)"
                    " VALUES (%s, %s, %s, %s, %s)",
                    (owner_id, member_id, STATUS_PENDING, relation, message),
                )
                link_id = int(cur.lastrowid)

            cur.execute(
                "SELECT l.*, u.id AS user_id, u.account, u.gender, u.age, l.created_at AS linked_at"
                " FROM family_links l JOIN users u ON u.id = l.member_id"
                " WHERE l.id = %s",
                (link_id,),
            )
            row = cur.fetchone()
    except FamilyError:
        raise
    except Exception as exc:  # noqa: BLE001
        logger.warning("family invite failed: %s", exc)
        raise DatabaseError("发送关联请求失败，请稍后再试。") from exc

    if not row:
        raise DatabaseError("关联请求没有保存成功，请重试。")
    return _row_to_link(row)


def incoming(member_id: int) -> List[Dict]:
    """我收到的、还没处理的关联请求（newest first）。"""
    try:
        with _cursor() as cur:
            cur.execute(
                "SELECT l.*, u.id AS user_id, u.account, u.gender, u.age, l.created_at AS linked_at"
                " FROM family_links l JOIN users u ON u.id = l.owner_id"
                " WHERE l.member_id = %s AND l.status = %s"
                " ORDER BY l.created_at DESC, l.id DESC",
                (int(member_id), STATUS_PENDING),
            )
            rows = cur.fetchall() or []
    except FamilyError:
        raise
    except Exception as exc:  # noqa: BLE001
        logger.warning("family incoming failed: %s", exc)
        raise DatabaseError("读取关联请求失败，请稍后再试。") from exc
    return [_row_to_link(row, perspective="member") for row in rows]


def incoming_count(member_id: int) -> int:
    try:
        with _cursor() as cur:
            cur.execute(
                "SELECT COUNT(*) AS n FROM family_links"
                " WHERE member_id = %s AND status = %s",
                (int(member_id), STATUS_PENDING),
            )
            row = cur.fetchone() or {}
        return int(row.get("n") or 0)
    except FamilyError:
        raise
    except Exception as exc:  # noqa: BLE001
        logger.warning("family incoming_count failed: %s", exc)
        raise DatabaseError("读取关联请求失败，请稍后再试。") from exc


def outgoing(owner_id: int) -> List[Dict]:
    """我发出的关联（pending / active / rejected 都返回，界面自己分组）。"""
    try:
        with _cursor() as cur:
            cur.execute(
                "SELECT l.*, u.id AS user_id, u.account, u.gender, u.age, l.created_at AS linked_at"
                " FROM family_links l JOIN users u ON u.id = l.member_id"
                " WHERE l.owner_id = %s"
                " ORDER BY l.created_at DESC, l.id DESC",
                (int(owner_id),),
            )
            rows = cur.fetchall() or []
    except FamilyError:
        raise
    except Exception as exc:  # noqa: BLE001
        logger.warning("family outgoing failed: %s", exc)
        raise DatabaseError("读取家庭组关联失败，请稍后再试。") from exc
    return [_row_to_link(row) for row in rows]


def active_members(owner_id: int) -> List[Dict]:
    """主账号已经获得授权的成员。"""
    try:
        with _cursor() as cur:
            cur.execute(
                "SELECT l.*, u.id AS user_id, u.account, u.gender, u.age, l.responded_at AS linked_at"
                " FROM family_links l JOIN users u ON u.id = l.member_id"
                " WHERE l.owner_id = %s AND l.status = %s"
                " ORDER BY l.responded_at DESC, l.id DESC",
                (int(owner_id), STATUS_ACTIVE),
            )
            rows = cur.fetchall() or []
    except FamilyError:
        raise
    except Exception as exc:  # noqa: BLE001
        logger.warning("family active_members failed: %s", exc)
        raise DatabaseError("读取家人列表失败，请稍后再试。") from exc
    return [_row_to_link(row) for row in rows]


def active_owners(member_id: int) -> List[Dict]:
    """已经获得我授权的账号（成员视角，可随时解除）。"""
    try:
        with _cursor() as cur:
            cur.execute(
                "SELECT l.*, u.id AS user_id, u.account, u.gender, u.age, l.responded_at AS linked_at"
                " FROM family_links l JOIN users u ON u.id = l.owner_id"
                " WHERE l.member_id = %s AND l.status = %s"
                " ORDER BY l.responded_at DESC, l.id DESC",
                (int(member_id), STATUS_ACTIVE),
            )
            rows = cur.fetchall() or []
    except FamilyError:
        raise
    except Exception as exc:  # noqa: BLE001
        logger.warning("family active_owners failed: %s", exc)
        raise DatabaseError("读取授权列表失败，请稍后再试。") from exc
    return [_row_to_link(row, perspective="member") for row in rows]


def respond(member_id: int, link_id: int, accept: bool) -> None:
    """被邀请账号接受 / 拒绝请求；只有 pending 的请求能被处理一次。

    接受前再校验一次「一个账号只属于一个家庭组」：对方可能已经被别人
    接受，或者管理者自己后来加入了别的家庭组。
    """
    status = STATUS_ACTIVE if accept else STATUS_REJECTED
    try:
        with _cursor() as cur:
            cur.execute(
                "SELECT owner_id, member_id, status FROM family_links WHERE id = %s",
                (int(link_id),),
            )
            link = cur.fetchone()
            if (
                not link
                or int(link["member_id"]) != int(member_id)
                or str(link["status"]) != STATUS_PENDING
            ):
                raise ValidationError("这条请求不存在，或已经处理过了。")

            if accept:
                member_group = _active_group_link(cur, int(member_id))
                if member_group:
                    raise ValidationError(
                        "你已经加入了一个家庭组，需先解除现有关系才能接受新的请求。"
                    )
                cur.execute(
                    "SELECT id FROM family_links"
                    " WHERE status = %s AND member_id = %s AND id <> %s LIMIT 1",
                    (STATUS_ACTIVE, int(link["owner_id"]), int(link_id)),
                )
                if cur.fetchone():
                    raise ValidationError(
                        "对方已经加入了其他家庭组，暂时无法接受这条请求。"
                    )

            cur.execute(
                "UPDATE family_links SET status = %s, responded_at = NOW()"
                " WHERE id = %s AND member_id = %s AND status = %s",
                (status, int(link_id), int(member_id), STATUS_PENDING),
            )
            if cur.rowcount == 0:
                raise ValidationError("这条请求不存在，或已经处理过了。")
    except FamilyError:
        raise
    except Exception as exc:  # noqa: BLE001
        logger.warning("family respond failed: %s", exc)
        raise DatabaseError("处理关联请求失败，请稍后再试。") from exc


def unlink(actor_id: int, link_id: int) -> None:
    """主账号移除成员、成员撤回授权、取消未处理请求，都走这里。

    只有关系里的 owner 或 member 本人能删；其他人拿到 link_id 也删不动。
    """
    actor_id = int(actor_id)
    try:
        with _cursor() as cur:
            cur.execute(
                "SELECT owner_id, member_id FROM family_links WHERE id = %s",
                (int(link_id),),
            )
            row = cur.fetchone()
            if not row:
                raise ValidationError("这条关联不存在，可能已经被对方解除。")
            if actor_id not in (int(row["owner_id"]), int(row["member_id"])):
                raise ValidationError("你没有权限解除这条关联。")
            cur.execute("DELETE FROM family_links WHERE id = %s", (int(link_id),))
    except FamilyError:
        raise
    except Exception as exc:  # noqa: BLE001
        logger.warning("family unlink failed: %s", exc)
        raise DatabaseError("解除关联失败，请稍后再试。") from exc


def group_owner_id(user_id: int) -> int:
    """这个账号所属家庭组的管理者 id；没有加入任何家庭组时就是自己。

    家庭组共享的数据（如家庭药箱）按这个 id 归属：管理者是自己，成员是
    发起邀请的 owner。一个账号只属于一个家庭组，取一条 active 就够。
    """
    user_id = int(user_id)
    try:
        with _cursor() as cur:
            row = _active_membership(cur, user_id)
    except FamilyError:
        raise
    except Exception as exc:  # noqa: BLE001
        logger.warning("family group_owner_id failed: %s", exc)
        raise DatabaseError("读取家庭组信息失败，请稍后再试。") from exc
    return int(row["owner_id"]) if row else user_id


def group_members(owner_id: int) -> List[Dict]:
    """家庭组的全部参与账号：管理者本人 + 所有 active 成员。

    用于「这个药箱共享给谁」「提醒发给谁」这类组级功能；没有家庭组时
    只返回自己，调用方不必区分是个人还是家庭组。
    """
    owner_id = int(owner_id)
    try:
        with _cursor() as cur:
            cur.execute(
                "SELECT id, account, gender, age FROM users WHERE id = %s",
                (owner_id,),
            )
            owner = cur.fetchone()
            cur.execute(
                "SELECT l.*, u.id AS user_id, u.account, u.gender, u.age,"
                " l.responded_at AS linked_at"
                " FROM family_links l JOIN users u ON u.id = l.member_id"
                " WHERE l.owner_id = %s AND l.status = %s"
                " ORDER BY l.responded_at ASC, l.id ASC",
                (owner_id, STATUS_ACTIVE),
            )
            rows = cur.fetchall() or []
    except FamilyError:
        raise
    except Exception as exc:  # noqa: BLE001
        logger.warning("family group_members failed: %s", exc)
        raise DatabaseError("读取家庭组成员失败，请稍后再试。") from exc

    members: List[Dict] = []
    if owner:
        members.append(
            {
                "id": int(owner["id"]),
                "user_id": int(owner["id"]),
                "account": str(owner.get("account") or ""),
                "gender": str(owner.get("gender") or ""),
                "age": int(owner["age"]) if owner.get("age") is not None else 0,
                "relation_title": "我",
                "is_owner": True,
            }
        )
    for row in rows:
        link = _row_to_link(row)
        link["is_owner"] = False
        members.append(link)
    return members


def authorized_member(owner_id: int, member_id: int) -> Optional[Dict]:
    """owner 对 member 是否有 active 授权；有则返回 member 的资料和 link 信息。"""
    try:
        with _cursor() as cur:
            cur.execute(
                "SELECT l.id, l.owner_id, l.member_id, l.status, l.relation,"
                " l.created_at, l.responded_at, u.id AS user_id, u.account,"
                " u.gender, u.age, l.responded_at AS linked_at"
                " FROM family_links l JOIN users u ON u.id = l.member_id"
                " WHERE l.owner_id = %s AND l.member_id = %s AND l.status = %s",
                (int(owner_id), int(member_id), STATUS_ACTIVE),
            )
            row = cur.fetchone()
    except FamilyError:
        raise
    except Exception as exc:  # noqa: BLE001
        logger.warning("family authorized_member failed: %s", exc)
        raise DatabaseError("校验家庭组权限失败，请稍后再试。") from exc
    return _row_to_link(row) if row else None
