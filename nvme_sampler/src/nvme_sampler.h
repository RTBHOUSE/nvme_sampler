typedef void *handle;

handle init_sampler(THFloatTensor *buffer,
                    const char *file_path,
                    long num_rows,
                    long row_size,
                    long max_batch_elements,
                    long max_num_threads,
                    long memory_usage_limit_b,
                    int seed);

void destroy_sampler(handle sampler);

long read_batch(handle sampler, long batch_size);
