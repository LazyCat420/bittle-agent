#!/bin/bash
# Reach the trainer (this PC, :8009) from the NAS-hosted bittle-agent when the Windows firewall
# blocks inbound 8009 (WSL2 mirrored networking; the Hyper-V firewall rule needs an admin shell).
#
#   PC :8009  --ssh -R-->  NAS 127.0.0.1:8029  --socat (container bittle-trainer-relay)-->  10.0.0.16:8039
#
# The bittle-agent container then uses BITTLE_TRAINER_URL=http://10.0.0.16:8039. One-time NAS setup:
#   sudo docker run -d --name bittle-trainer-relay --restart unless-stopped --network host \
#        alpine/socat TCP-LISTEN:8039,bind=10.0.0.16,fork,reuseaddr TCP:127.0.0.1:8029
#   and `Match User lazycat / AllowTcpForwarding yes` in /etc/ssh/sshd_config (DSM defaults to no).
# Run:  setsid nohup trainer/scripts/nas_tunnel.sh > runs/nas_tunnel.log 2>&1 < /dev/null &
# The proper fix is an inbound rule for TCP 8009 on the Windows host (elevated PowerShell):
#   New-NetFirewallHyperVRule -Name bittle-trainer-8009 -DisplayName 'bittle trainer 8009' -Direction Inbound \
#       -VMCreatorId '{40E0AC32-46A5-438A-A0B2-2B479E8F2E90}' -Protocol TCP -LocalPorts 8009 -Action Allow
#   New-NetFirewallRule -DisplayName 'bittle-trainer 8009' -Direction Inbound -Protocol TCP -LocalPort 8009 -Action Allow
# after which BITTLE_TRAINER_URL can go back to http://10.0.0.171:8009 and this tunnel is not needed.
NAS="${NAS_SSH_HOST:-nas}"
while true; do
  ssh -N -o ControlMaster=no -o ControlPath=none -o ServerAliveInterval=30 -o ServerAliveCountMax=3 \
      -o ExitOnForwardFailure=yes -R 127.0.0.1:8029:127.0.0.1:8009 "$NAS"
  sleep 5
done
