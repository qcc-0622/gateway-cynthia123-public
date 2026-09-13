#!/usr/bin/env bash
# 一键部署（REFACTOR_ROADMAP P1.2）：测试 → push → 服务器漂移检查 → pull → 重启 → 健康检查
# 用法：bash scripts/deploy.sh
# 依赖：scripts/deploy.local（不入 git，含 SSH_HOST/SSH_PORT/SSH_KEY 三个变量）
set -euo pipefail
cd "$(dirname "$0")/.."

LOCAL_CONF="scripts/deploy.local"
if [[ ! -f "$LOCAL_CONF" ]]; then
    echo "❌ 缺少 $LOCAL_CONF（不入 git 的本地配置）。请创建，内容示例："
    echo '   SSH_HOST=ubuntu@1.2.3.4'
    echo '   SSH_PORT=22'
    echo '   SSH_KEY=$HOME/.ssh/id_ed25519'
    exit 1
fi
# shellcheck disable=SC1090
source "$LOCAL_CONF"
SSH_CMD="ssh -p ${SSH_PORT} -i ${SSH_KEY} -o ConnectTimeout=10 -o StrictHostKeyChecking=no ${SSH_HOST}"

echo "== 1/5 本地测试 =="
# 顺序跑 tests/ 下所有测试文件（各自支持 python 直跑，无需 pytest）
rm -f /tmp/gw_test_out.txt
for t in tests/test_*.py; do
    python "$t" >> /tmp/gw_test_out.txt 2>&1 || { tail -5 /tmp/gw_test_out.txt; echo "❌ $t 测试未通过，部署中止"; exit 1; }
    echo "  PASS $(basename "$t")"
done

echo "== 2/5 git push =="
git push

echo "== 3/5 服务器漂移检查 =="
DRIFT_ALL=$($SSH_CMD "cd /opt/chat_gateway && git status --porcelain")
DRIFT_TRACKED=$(echo "$DRIFT_ALL" | grep -v '^??' | grep -v '^$' || true)
DRIFT_UNTRACKED=$(echo "$DRIFT_ALL" | grep '^??' || true)
if [[ -n "$DRIFT_TRACKED" ]]; then
    echo "❌ 服务器有**已跟踪文件被改动**（真漂移），部署中止。先收编进 git 或 stash："
    echo "$DRIFT_TRACKED" | head -10
    exit 1
fi
if [[ -n "$DRIFT_UNTRACKED" ]]; then
    echo "⚠️ 服务器有未跟踪散落文件（不阻塞部署，但该归档了，见 ROADMAP P4.3）："
    echo "$DRIFT_UNTRACKED" | head -10
fi

echo "== 4/5 服务器 pull + 重启 =="
$SSH_CMD "cd /opt/chat_gateway && git pull --ff-only && sudo systemctl restart chat-gateway && sleep 3 && systemctl is-active chat-gateway"

echo "== 5/5 健康检查 =="
$SSH_CMD "curl -s -m 5 http://127.0.0.1:8899/health && echo && journalctl -u chat-gateway --since '30 seconds ago' --no-pager | grep -ciE 'error|exception|traceback' | xargs -I{} echo '启动日志报错行数: {}'"

echo "✅ 部署完成"
