.PHONY: lib

all: lib

lib:
	$(MAKE) -C lib

clean:
	$(MAKE) clean -C lib
	rm -rf nvme_sampler/_ext

test:
	python3 -m pytest --capture=no -vv

test_perf:
	./lib/bin/test_perf_memcpy 128 16 16 1073741824 4
	./lib/bin/test_perf_memcpy 128 32 32 1073741824 4
	./lib/bin/test_perf_memcpy 1024 16 16 1073741824 4
	./lib/bin/test_perf_memcpy 1024 32 32 1073741824 4
	./lib/bin/test_perf_memcpy 4096 16 16 1073741824 4
	./lib/bin/test_perf_memcpy 4096 32 32 1073741824 4

	#./lib/tests/create_sample_file.sh

	./lib/bin/test_perf_nvme /mnt/ssd1/pwiejacha/nvme/100GiB.bin 1280 83886080 16384 16 10737418240
