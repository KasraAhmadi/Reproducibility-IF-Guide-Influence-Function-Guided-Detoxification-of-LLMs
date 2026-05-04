#!/bin/bash

PROJECT="personality-embedding"
MACHINE_TYPE="n1-standard-2"
GPU_TYPE="nvidia-tesla-t4"
INSTANCE_NAME="gpu-machine"
SERVICE_ACCOUNT="FILL_WITH_YOUR_SERVICE_ACCOUNT_EMAIL"
SNAPSHOT="FILL_WITH_YOUR_SNAPSHOT_NAME"

echo "🔍 Searching for available T4 GPUs across all zones..."

ZONES=$(gcloud compute accelerator-types list \
  --filter="name=${GPU_TYPE}" \
  --format="value(zone)" \
  --project="${PROJECT}" 2>/dev/null | sort)

if [ -z "$ZONES" ]; then
  echo "❌ No zones found with ${GPU_TYPE}"
  exit 1
fi

echo "Zones with T4: $(echo $ZONES | tr '\n' ' ')"
echo ""

for ZONE in $ZONES; do
  REGION=$(echo "$ZONE" | sed 's/-[a-z]$//')
  echo "⏳ Trying zone: ${ZONE} (region: ${REGION})..."

  OUTPUT=$(gcloud compute instances create "${INSTANCE_NAME}" \
    --project="${PROJECT}" \
    --zone="${ZONE}" \
    --machine-type="${MACHINE_TYPE}" \
    --network-interface=network-tier=PREMIUM,stack-type=IPV4_ONLY,subnet=default \
    --maintenance-policy=TERMINATE \
    --provisioning-model=STANDARD \
    --service-account="${SERVICE_ACCOUNT}" \
    --scopes=https://www.googleapis.com/auth/devstorage.read_only,https://www.googleapis.com/auth/logging.write,https://www.googleapis.com/auth/monitoring.write,https://www.googleapis.com/auth/service.management.readonly,https://www.googleapis.com/auth/servicecontrol,https://www.googleapis.com/auth/trace.append \
    --accelerator=count=1,type="${GPU_TYPE}" \
    --create-disk=auto-delete=yes,boot=yes,device-name="${INSTANCE_NAME}",mode=rw,size=300,source-snapshot="${SNAPSHOT}",type=pd-balanced \
    --no-shielded-secure-boot \
    --shielded-vtpm \
    --shielded-integrity-monitoring \
    --labels=goog-ec-src=vm_add-gcloud \
    --reservation-affinity=any 2>&1)

  if echo "$OUTPUT" | grep -q "RUNNING\|created"; then
    echo ""
    echo "✅ SUCCESS! Instance created in zone: ${ZONE}"
    echo "Instance name: ${INSTANCE_NAME}"
    echo "$OUTPUT"
    exit 0
  else
    REASON=$(echo "$OUTPUT" | grep -i "error\|quota\|resource\|capacity" | head -1)
    echo "   ❌ Failed: ${REASON:-Unknown error}"
  fi

done

echo ""
echo "❌ No available zone found with a standard T4 GPU."
echo "Try again later or consider nvidia-l4 as an alternative."
exit 1
