oas process-units --shard 6 --n-shards 40 --n-jobs 8 --chunk-size 8000 &>shard-6-40-dgx3.log &
disown
oas process-units --shard 10 --n-shards 40 --n-jobs 8 --chunk-size 8000 &>shard-10-40-dgx3.log &
disown
oas process-units --shard 11 --n-shards 40 --n-jobs 8 --chunk-size 8000 &>shard-11-40-dgx3.log &
disown
oas process-units --shard 18 --n-shards 40 --n-jobs 8 --chunk-size 8000 &>shard-18-40-dgx3.log &
disown
oas process-units --shard 23 --n-shards 40 --n-jobs 8 --chunk-size 8000 &>shard-23-40-dgx3.log &
disown
oas process-units --shard 29 --n-shards 40 --n-jobs 8 --chunk-size 8000 &>shard-29-40-dgx3.log &
disown
oas process-units --shard 38 --n-shards 40 --n-jobs 8 --chunk-size 8000 &>shard-38-40-dgx3.log &
disown
oas process-units --shard 39 --n-shards 40 --n-jobs 8 --chunk-size 8000 &>shard-39-40-dgx3.log &
disown