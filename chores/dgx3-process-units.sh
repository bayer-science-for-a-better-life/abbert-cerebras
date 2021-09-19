oas process-units --shard 2 --n-shards 48 --n-jobs 8 --chunk-size 8000 &>shard-2-48-dgx3.log &
disown
oas process-units --shard 4 --n-shards 48 --n-jobs 8 --chunk-size 8000 &>shard-4-48-dgx3.log &
disown
oas process-units --shard 6 --n-shards 48 --n-jobs 8 --chunk-size 8000 &>shard-6-48-dgx3.log &
disown
oas process-units --shard 7 --n-shards 48 --n-jobs 8 --chunk-size 8000 &>shard-7-48-dgx3.log &
disown
oas process-units --shard 15 --n-shards 48 --n-jobs 8 --chunk-size 8000 &>shard-15-48-dgx3.log &
disown
oas process-units --shard 23 --n-shards 48 --n-jobs 8 --chunk-size 8000 &>shard-23-48-dgx3.log &
disown
oas process-units --shard 27 --n-shards 48 --n-jobs 8 --chunk-size 8000 &>shard-27-48-dgx3.log &
disown
oas process-units --shard 30 --n-shards 48 --n-jobs 8 --chunk-size 8000 &>shard-30-48-dgx3.log &
disown
oas process-units --shard 45 --n-shards 48 --n-jobs 8 --chunk-size 8000 &>shard-45-48-dgx3.log &
disown
oas process-units --shard 47 --n-shards 48 --n-jobs 8 --chunk-size 8000 &>shard-47-48-dgx3.log &
disown