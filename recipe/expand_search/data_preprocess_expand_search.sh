python -m ferret.data.preprocess \
    --template expand_search \
    --train_data_sources nq,hotpotqa \
    --test_data_sources nq,triviaqa,popqa,hotpotqa,2wikimultihopqa,musique,bamboogle \
    --local_dir data \
    --test_subset_ratio 0.1
