python -m scout.data.preprocess \
    --template parallel_search \
    --train_data_sources nq,hotpotqa \
    --test_data_sources nq,triviaqa,popqa,hotpotqa,2wikimultihopqa,musique,bamboogle \
    --local_dir data
