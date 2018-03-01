#!/bin/bash

set -eo pipefail

echo Creating random 100 GiB file. This may take a while.

test -n "$NVME_WORKDIR" || { echo "$NVME_WORKDIR not set aborting"; exit 1 }

dd if=/dev/urandom of=$NVME_WORKDIR/random_sample count=4096 bs=1M

for i in `seq 25`; do
	{ cat $NVME_WORKDIR/random_sample | shuf >> $NVME_WORKDIR/100GiB.bin ; sleep 2; } &
done

dd if=$NVME_WORKDIR/100GiB.bin of=$NVME_WORKDIR/100GiB_seq.bin bs=4M

