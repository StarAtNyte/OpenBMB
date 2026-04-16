# CloudRift Setup for SOAR FP4 Optimization

## Prerequisites
- CloudRift account with positive balance
- SSH client (PowerShell on Windows works)
- Local copy of `soar-optimization/submission/`

---

## Part 1: Create Instance (One Time)

### 1.1 Create Persistent Volume First
1. Go to [cloudrift.ai](https://cloudrift.ai) → **Volumes** tab
2. Click **Create Volume**
3. Name: `soar-workspace`
4. Size: **150 GB**
5. Select datacenter with **RTX PRO 6000** support
6. Click **Create**

### 1.2 Launch Docker Instance
1. Go to **Instances** → **New Instance**
2. Select **Docker** (not VM)
3. GPU: **RTX PRO 6000** (must show "Persistent storage support")
4. Docker Image: `ghcr.io/openbmb/soar-toolkit:latest`
5. Attach Volume: select `soar-workspace` → mount path `/workspace`
6. Expose Port: `30000`
7. Click **Deploy**

---

## Part 2: First-Time Setup (~50 min, do once)

### 2.1 Connect
```bash
ssh riftuser@<ip> -p <port>
```

### 2.2 Mount Volume
```bash
# Check volume appeared
lsblk
# You should see vdb (or similar) — the size you chose

# Format ONLY on first use — never again or all data is wiped
sudo mkfs.ext4 /dev/vdb

# Mount
sudo mkdir -p /workspace
sudo mount /dev/vdb /workspace
sudo chown -R riftuser:riftuser /workspace

# Auto-mount on reboot
echo '/dev/vdb /workspace ext4 defaults 0 2' | sudo tee -a /etc/fstab
```

> ⚠️ **Never run `mkfs.ext4` again after this step**

### 2.3 Verify Model is Present
```bash
ls /models/MiniCPM-SALA/
# Should show model shards, config.json, tokenizer files etc.
```

### 2.4 Upload Submission Files
Run this **on your local Windows machine** (PowerShell):
```powershell
scp -r -P <port> D:\Projects\Others\OpenBMB\soar-optimization\submission\ riftuser@<ip>:/workspace/submission/
```

### 2.5 Patch SGLang
```bash
bash /workspace/submission/prepare_env.sh
# Should print: patched + recompiled: srt/layers/linear.py etc.
```

### 2.6 Quantize Model (~45 min)
```bash
bash /workspace/submission/prepare_model.sh \
  --input /models/MiniCPM-SALA \
  --output /workspace/nvfp4_model
# Saves quantized model to persistent volume
```

> ✅ **After this completes: stop the instance to save money**
> The quantized model is saved and never needs to be recreated unless you change `preprocess_model_nvfp4_modelopt.py`

---

## Part 3: Every Session

### 3.1 Start Instance
1. Go to CloudRift dashboard → **Instances**
2. Click **Start** on your stopped instance
3. Wait for it to become **Running**

### 3.2 Connect & Mount
```bash
ssh riftuser@<ip> -p <port>

# Mount volume (/etc/fstab handles this automatically if set up above)
# If not auto-mounted:
sudo mount /dev/vdb /workspace

# Verify quantized model is there
ls /workspace/nvfp4_model/
```

### 3.3 Re-Patch SGLang
```bash
# /opt is part of Docker image — patches are wiped on restart
bash /workspace/submission/prepare_env.sh
```

### 3.4 Start Server
```bash
python3 -m sglang.launch_server \
  --log-level info \
  --attention-backend minicpm_flashinfer \
  --chunked-prefill-size 8192 \
  --mem-fraction-static 0.70 \
  --dense-as-sparse \
  --skip-server-warmup \
  --quantization modelopt_fp4 \
  --disable-radix-cache \
  --tokenizer-path /models/MiniCPM-SALA \
  --model-path /workspace/nvfp4_model
```

Wait for:
```
The server is fired up and ready to roll!
```

---

## Part 4: Iterating on Code Changes

### Only SGLang patches changed (linear.py, minicpm.py, etc.)
```powershell
# Local — re-upload submission
scp -r -P <port> D:\Projects\Others\OpenBMB\soar-optimization\submission\ riftuser@<ip>:/workspace/submission/
```
```bash
# Instance — re-patch and restart server
bash /workspace/submission/prepare_env.sh
# restart server command from Part 3.4
```

### Quantization script changed (preprocess_model_nvfp4_modelopt.py)
```powershell
# Local — re-upload
scp -r -P <port> D:\Projects\Others\OpenBMB\soar-optimization\submission\ riftuser@<ip>:/workspace/submission/
```
```bash
# Instance — re-quantize (overwrites previous model)
bash /workspace/submission/prepare_env.sh
bash /workspace/submission/prepare_model.sh \
  --input /models/MiniCPM-SALA \
  --output /workspace/nvfp4_model
```

---

## Storage Layout

| Path | Stored On | Survives Restart |
|------|-----------|-----------------|
| `/workspace/nvfp4_model/` | Persistent volume | ✅ Yes |
| `/workspace/submission/` | Persistent volume | ✅ Yes |
| `/models/MiniCPM-SALA/` | Docker image | ✅ Yes |
| `/opt/SGLang-MiniCPM-SALA/` | Docker image | ⚠️ Patches wiped — re-run `prepare_env.sh` |
| `/opt/SOAR/` | Docker image | ✅ Yes |

---

## Cost Tips

| Action | Impact |
|--------|--------|
| Stop instance when not testing | GPU billed per hour only when running |
| Keep account balance positive | CloudRift **deletes volumes** at zero balance — top up before running out |
| Never delete `/workspace/nvfp4_model` | Costs 45 min GPU time to recreate |
| Re-quantize only when `preprocess_model_nvfp4_modelopt.py` changes | SGLang patch changes don't need re-quantization |
