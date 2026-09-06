import pytest
from unittest.mock import AsyncMock, patch

from mcp.client import TOOLS, execute


def test_mcp_tools_dispatch_table_completeness():
    required_tools = [
        "bittle_list_capabilities",
        "bittle_status",
        "bittle_do_skill",
        "bittle_move_joints",
        "bittle_estop",
        "bittle_execute_sequence",
        "bittle_save_moveset",
        "bittle_list_primitives",
        "bittle_compose_move",
        "bittle_iterate_move",
        "bittle_evaluate_move",
        "bittle_list_library",
        "bittle_load_moveset",
    ]
    for t in required_tools:
        assert t in TOOLS, f"Missing tool in TOOLS dispatch: {t}"


@pytest.mark.anyio
async def test_execute_unknown_tool():
    res = await execute("non_existent_tool", {})
    assert res["ok"] is False
    assert res["error"] == "unknown_tool"


@pytest.mark.anyio
async def test_execute_list_primitives_dispatch():
    with patch("mcp.client._client.list_primitives", new_callable=AsyncMock) as mock_list:
        mock_list.return_value = {"ok": True, "primitives": []}
        res = await execute("bittle_list_primitives", {"group": "head"})
        assert res["ok"] is True
        mock_list.assert_called_once_with(group="head")


@pytest.mark.anyio
async def test_execute_compose_move_dispatch():
    with patch("mcp.client._client.compose_move", new_callable=AsyncMock) as mock_compose:
        mock_compose.return_value = {"ok": True, "moveset": {"name": "combo"}}
        res = await execute("bittle_compose_move", {
            "name": "combo",
            "primitives": ["head_scan_left"],
            "mode": "sequential",
        })
        assert res["ok"] is True
        mock_compose.assert_called_once_with(
            "combo",
            ["head_scan_left"],
            mode="sequential",
            description="",
        )


@pytest.mark.anyio
async def test_execute_iterate_move_dispatch():
    with patch("mcp.client._client.iterate_move", new_callable=AsyncMock) as mock_iter:
        mock_iter.return_value = {"ok": True, "moveset": {"name": "combo"}}
        res = await execute("bittle_iterate_move", {
            "name": "combo",
            "adjustments": {"delay_scale": 1.2},
        })
        assert res["ok"] is True
        mock_iter.assert_called_once_with("combo", {"delay_scale": 1.2})
