oas process-units --shard 0 --n-shards 10 --n-jobs 8 --chunk-size 8000 &>shard-0-10-dgx2.log &
disown
oas process-units --shard 1 --n-shards 10 --n-jobs 8 --chunk-size 8000 &>shard-1-10-dgx2.log &
disown
oas process-units --shard 2 --n-shards 10 --n-jobs 8 --chunk-size 8000 &>shard-2-10-dgx2.log &
disown
oas process-units --shard 3 --n-shards 10 --n-jobs 8 --chunk-size 8000 &>shard-3-10-dgx2.log &
disown
oas process-units --shard 4 --n-shards 10 --n-jobs 8 --chunk-size 8000 &>shard-4-10-dgx2.log &
disown
oas process-units --shard 5 --n-shards 10 --n-jobs 8 --chunk-size 8000 &>shard-5-10-dgx2.log &
disown
oas process-units --shard 6 --n-shards 10 --n-jobs 8 --chunk-size 8000 &>shard-6-10-dgx2.log &
disown
oas process-units --shard 7 --n-shards 10 --n-jobs 8 --chunk-size 8000 &>shard-7-10-dgx2.log &
disown
oas process-units --shard 8 --n-shards 10 --n-jobs 8 --chunk-size 8000 &>shard-8-10-dgx2.log &
disown
oas process-units --shard 9 --n-shards 10 --n-jobs 8 --chunk-size 8000 &>shard-9-10-dgx2.log &
disown