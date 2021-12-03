#!/bin/bash

cd /cb/home/aarti/ws/code/bayer_tfrecs_filtering/collect_stats

cbrun -t cpu srund -x "-p cpu -C m5.12xlarge -c 48 -J bayerstats00 -o bayerstats00.log --time 8-00:00:00" -e "source create_parallel_stats.sh /cb/customers/bayer/new_datasets/filters_default/stats/filenames/bayer_filters_default_for_stats_00.txt"


cbrun -t cpu srund -x "-p cpu -C m5.12xlarge -c 48 -J bayerstats01 -o bayerstats01.log --time 8-00:00:00" -e "source create_parallel_stats.sh /cb/customers/bayer/new_datasets/filters_default/stats/filenames/bayer_filters_default_for_stats_01.txt"

cbrun -t cpu srund -x "-p cpu -C m5.12xlarge -c 48 -J bayerstats02 -o bayerstats02.log --time 8-00:00:00" -e "source create_parallel_stats.sh /cb/customers/bayer/new_datasets/filters_default/stats/filenames/bayer_filters_default_for_stats_02.txt"

cbrun -t cpu srund -x "-p cpu -C m5.12xlarge -c 48 -J bayerstats03 -o bayerstats03.log --time 8-00:00:00" -e "source create_parallel_stats.sh /cb/customers/bayer/new_datasets/filters_default/stats/filenames/bayer_filters_default_for_stats_03.txt"

cbrun -t cpu srund -x "-p cpu -C m5.12xlarge -c 48 -J bayerstats04 -o bayerstats04.log --time 8-00:00:00" -e "source create_parallel_stats.sh /cb/customers/bayer/new_datasets/filters_default/stats/filenames/bayer_filters_default_for_stats_04.txt"

cbrun -t cpu srund -x "-p cpu -C m5.12xlarge -c 48 -J bayerstats05 -o bayerstats05.log --time 8-00:00:00" -e "source create_parallel_stats.sh /cb/customers/bayer/new_datasets/filters_default/stats/filenames/bayer_filters_default_for_stats_05.txt"