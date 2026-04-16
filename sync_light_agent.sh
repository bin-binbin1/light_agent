#!/bin/bash
# 将 light_agent 源仓库的最新代码同步到 safety_score_agent/light_agent/
# 用法: bash sync_light_agent.sh
# 可放到 light_agent_source 仓库的 post-commit hook 或手动执行

set -e

# ── 路径配置（按实际修改） ──
SOURCE_DIR="C:\\code\\light_agent"
TARGET_DIR="C:\\code\\python\\safety_score_agent\\light_agent"

# ── 拉取最新代码 ──
echo "=== 拉取 light_agent 最新代码 ==="
cd "$SOURCE_DIR"
git pull origin main

# ── 同步到目标目录 ──
echo "=== 同步到 safety_score_agent/light_agent/ ==="
# 删除旧文件（排除可能的本地临时文件）
find "$TARGET_DIR" -mindepth 1 -not -name '__pycache__' -not -path '*/__pycache__/*' -delete 2>/dev/null || true

# 复制新文件，排除 .git / .github / __pycache__ / .idea
cd "$SOURCE_DIR"
find . -mindepth 1 \
  -not -path './.git/*' -not -name '.git' \
  -not -path './.github/*' -not -name '.github' \
  -not -path './__pycache__/*' -not -name '__pycache__' \
  -not -path './.idea/*' -not -name '.idea' \
  -type f -exec cp --parents {} "$TARGET_DIR/" \;

echo "=== 同步完成 ==="
echo "请到 safety_score_agent 目录检查并提交:"
echo "  cd $(dirname "$TARGET_DIR")"
echo "  git add light_agent/ && git commit -m 'sync: update light_agent'"
