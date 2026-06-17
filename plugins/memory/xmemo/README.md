# XMemo Memory Provider

User-owned cloud memory for AI agents. XMemo provides orchestrated recall,
semantic search, durable fact storage, working state, reminders, and session
snapshots across sessions and tools.

## Requirements

- Hermes already depends on `httpx`.
- XMemo service token from [xmemo.dev](https://xmemo.dev).

## Setup

```bash
hermes memory setup xmemo
```

Or manually:

```bash
hermes config set memory.provider xmemo
echo "XMEMO_KEY=your-token" >> ~/.hermes/.env
```

## Config

Config file: `$HERMES_HOME/xmemo.json`

| Key | Default | Description |
|-----|---------|-------------|
| `base_url` | `https://xmemo.dev` | XMemo service URL |
| `agent_id` | `hermes` | Agent family identifier |
| `agent_instance_id` | auto-generated | Stable device/install identifier |
| `bucket` | `work` | Storage namespace |
| `scope` | `hermes/default` | Project/session scope |
| `timeout_seconds` | `5.0` | REST request timeout |
| `prefetch_max_items` | `5` | Max context items per recall |
| `prefetch_max_tokens` | `900` | Max context tokens per recall |

## Tools

| Tool | Description |
|------|-------------|
| `xmemo_search` | Semantic search over XMemo memories |
| `xmemo_remember` | Save a durable fact |
| `xmemo_update_state` | Save active task / next action / blocker with TTL |
| `xmemo_recall_context` | Build a bounded, ranked context pack |
| `xmemo_record_event` | Append a timeline event or milestone |
| `xmemo_create_reminder` | Create a TODO / action item |
| `xmemo_list_reminders` | List open or completed reminders |
| `xmemo_complete_reminder` | Mark a reminder as completed |
| `xmemo_mark_used` | Record that a recalled memory was used |
| `xmemo_forget` | Delete a memory by id or "current" |
