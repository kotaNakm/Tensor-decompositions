cd `dirname $0`
cd ..

INPUT="gowalla"
input_type="AGH_experiment1"
entities="userid/placeid/yr_month"
value_column="checkins"

tag="test"
train_ratio=0.7
gamma=0.1
n_iter=20
rank=5
l0=1e-100


# if true;then
if false;then
OUTPUT="_out/"$INPUT"/"$input_type"/train_ratio_"$train_ratio"/rank_"$rank"/gamma_"$gamma"/l0_"$l0"/niter_"$n_iter"/tag_"$tag
python3 factorization/agh/main.py    --input_tag $INPUT \
                            --input_type $input_type \
                            --out_dir $OUTPUT \
                            --entities $entities \
                            --value_column $value_column \
                            --train_ratio $train_ratio \
                            --rank $rank \
                            --initial_gamma $gamma \
                            --n_iter $n_iter \
                            --l0 $l0 \
                            # 
fi


if true;then
# if false;then
OUTPUT="_out/"$INPUT"/"$input_type"/train_ratio_"$train_ratio"/rank_"$rank"/niter_"$n_iter"/tag_"$tag
python3 factorization/parafac/main.py    --input_tag $INPUT \
                            --input_type $input_type \
                            --out_dir $OUTPUT \
                            --entities $entities \
                            --value_column $value_column \
                            --train_ratio $train_ratio \
                            --rank $rank \
                            --n_iter $n_iter \
                            # 
fi