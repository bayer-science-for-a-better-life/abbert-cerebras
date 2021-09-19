oas process-units --shard 0 --n-shards 40 --n-jobs 8 --chunk-size 8000 &>shard-0-40-dgx2.log &
disown
oas process-units --shard 2 --n-shards 40 --n-jobs 8 --chunk-size 8000 &>shard-2-40-dgx2.log &
disown
oas process-units --shard 5 --n-shards 40 --n-jobs 8 --chunk-size 8000 &>shard-5-40-dgx2.log &
disown
oas process-units --shard 7 --n-shards 40 --n-jobs 8 --chunk-size 8000 &>shard-7-40-dgx2.log &
disown
oas process-units --shard 8 --n-shards 40 --n-jobs 8 --chunk-size 8000 &>shard-8-40-dgx2.log &
disown
oas process-units --shard 9 --n-shards 40 --n-jobs 8 --chunk-size 8000 &>shard-9-40-dgx2.log &
disown
oas process-units --shard 16 --n-shards 40 --n-jobs 8 --chunk-size 8000 &>shard-16-40-dgx2.log &
disown
oas process-units --shard 34 --n-shards 40 --n-jobs 8 --chunk-size 8000 &>shard-34-40-dgx2.log &
disown