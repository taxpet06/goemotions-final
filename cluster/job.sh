# Runs ON the GPU node. Code+data arrive as a tar on stdin, outputs leave as a tar on
# stdout (GPU nodes cannot read the cluster home; see README "Running on Discovery").
set -uo pipefail
exec 3>&1 1>&2
W=/tmp/ge_${SLURM_JOB_ID}; mkdir -p "$W"; cd "$W"; tar xzf -
export HOME=$W XDG_CACHE_HOME=$W/cache HF_HOME=$W/hf UV_CACHE_DIR=$W/uvc PIP_CACHE_DIR=$W/pipc
echo "node $(hostname) $(nvidia-smi -L)"
python3 -m venv venv && venv/bin/pip -q install uv
venv/bin/uv pip install -q --python venv/bin/python --extra-index-url https://download.pytorch.org/whl/cu126 \
  torch==2.7.1 transformers==4.46.3 numpy==2.2.6
if grep -q bilstm "runs/$JOB.sh"; then
  curl -sSfL -o g.zip https://nlp.stanford.edu/data/glove.6B.zip || curl -sSfL -o g.zip https://huggingface.co/stanfordnlp/glove/resolve/main/glove.6B.zip
  python3 -c "import zipfile,shutil; shutil.copyfileobj(zipfile.ZipFile('g.zip').open('glove.6B.300d.txt'), open('glove.6B.300d.txt','wb'))"; rm g.zip
fi
PATH=$W/venv/bin:$PATH bash "runs/$JOB.sh" || echo "SOME RUN FAILED"
tar czf - outputs >&3
cd /; rm -rf "$W"
