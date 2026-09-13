# Chat Gateway

一个自托管的 **AI API 中转网关**。它夹在你日常用的客户端（RikkaHub 之类的
Anthropic 协议 App）和上游 API 中转站之间，专门服务"长期陪伴型对话"场景——
目标是让 AI 在超长对话里不失忆，同时把 token 账单压到最低。

> 这个仓库是个人自用网关的**脱敏副本**：私有部署信息、密钥和个人规划文档已剥离，
> 代码逻辑本身未改动。供参考借鉴。

## 它解决什么问题

| 痛点 | 做法 |
|------|------|
| 长对话越来越贵 | **BP1-BP4 缓存断点**：自动在 `tools → system → messages` 上注入 `cache_control: ephemeral`（Anthropic 上限 4 个），最大化 Prompt Cache 命中 |
| 上下文塞不下 | **BP2 滚动摘要**：超阈值的历史压缩成分层摘要（锚定 / 近期 / 远期 / 古老），而不是粗暴截断丢掉 |
| 换新窗口就失忆 | **Seamless Session**：新会话自动注入旧摘要 + 归档记录 |
| 想要长期记忆 | **Memory Recall**：接 Ombre 记忆库，新话题时做语义检索并自动召回/回喂 |
| 多家中转站模型与价格不一 | **多上游路由**：`upstream::model` 前缀精确路由 + 多令牌分组（按用途 / 模型 / 前缀 / cache TTL 分流） |
| 花了多少钱不清楚 | **缓存监控**：token / 缓存命中 / 费用按「中转站 × 模型」统计；**上游价格自动同步**——直接读 new-api 的 `/api/pricing` 换算单价，不用手填价格表 |
| 重新生成会串味 | **Re-roll 去重**：客户端点重新生成时自动清掉上一轮的记录，避免账本分叉 |
| 想看历史在聊什么 | **管理后台**：暗色模式 / PWA / 移动端适配 / 通知铃铛 / 对话日志与摘要管理 |

## 快速开始

```bash
git clone <repo-url>
cd chat-gateway

# 1. 配置
cp .env.example .env                                # 改密钥/端口/时区
cp data/settings.example.json data/settings.json    # 模型与功能开关
cp data/upstreams.example.json data/upstreams.json  # 填上游 API key

# 2. 依赖
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt

# 3. 启动
python run.py
```

监听 `127.0.0.1:8899`，通常由 Nginx 反代到 `https://your.domain/` 并挂到
`/chat-gateway/` 子路径下（`gateway/config.py` 与模板里的 `admin_base` 按需调整）。

## 上游价格自动同步（new-api）

上游如果是 [new-api](https://github.com/QuantumNous/new-api) 中转站，它的
`GET /api/pricing` 会返回每个模型的全部计费倍率，网关可以据此自动算出单价：

- **换算式**：`$/1M 输入 = model_ratio × group_ratio × 2`（new-api 里 `QuotaPerUnit = 500000`，
  即倍率 1 = $0.002/1K）；输出乘 `completion_ratio`；缓存读乘 `cache_ratio`；
  缓存写乘 `create_cache_ratio`（1h TTL 再 ×1.6）；`quota_type=1` 的模型走按次价 `model_price × group_ratio`。
- **取价优先级**：手工价格表 > 上游自动同步价 > 内置默认价。手工永远优先，
  想让某个模型走手填值就加一行，不用关自动同步。
- **开关位置**：管理后台 → 网关设置 → 价格表。
- **两个坑**（详见 `gateway/services/price_sync.py` 模块注释）：
  ① 拉 `/api/pricing` **不能带 Authorization 头**——那个接口只认 new-api 的后台凭据，
     带上转发用的 key 会被判成坏凭据并直接 403；
  ② `cache_ratio` / `create_cache_ratio` 是 `*float64 + omitempty`，
     **字段缺失 ≠ 0**（缺失表示站点没配过，要用默认值 1.0 / 1.25）。

## 目录结构

```
gateway/
├── main.py            FastAPI 应用 + lifespan（后台任务：keepalive / 余额 / 价格同步）
├── hooks.py           预处理管道入口
├── pipeline/          声明式管道（BP1-BP4 / Seamless / Tag / 主动消息 / 摘要 / 统一大脑）
├── services/          归档、转发、模型列表同步、价格同步
├── routers/           anthropic / openai / models / admin 四组路由
├── memory.py          Ombre 记忆召回
├── summarizer.py      BP2 滚动摘要
├── notifier.py        推送（PushPlus / ntfy / Web Push）
├── db.py              SQLite 读写
├── costs.py           费用计算（取价唯一真相源）
├── upstream.py        上游配置与路由
├── settings.py        运行时设置（data/settings.json）
└── templates/         管理后台 Jinja2 页面
tests/                 回归测试（逐文件可直接 python 运行）
docs/architecture.md   整体架构
CHAT上下文与缓存策略.md   上下文识别与缓存断点策略
```

## 测试

```bash
python -m pytest tests/          # 全部
python tests/test_failover.py    # 单文件直跑也支持（无需 pytest）
```

`scripts/deploy.sh` 是一键部署脚本（本地测试 → push → 服务器漂移检查 → pull →
重启 → 健康检查），服务器地址从 gitignore 掉的 `scripts/deploy.local` 读，可以照着改。

## License

MIT —— 见 [LICENSE](LICENSE)。版权人写的是 GitHub 用户名，不含真实姓名。
