cd `dirname $0`
cd ..

INPUT="synthetic"
input_type="10,10,20"
entities="entity1/entity2/entity3"
value_column="value"

train_ratio=0.7


# if true;then
if false;then
tag="test"
gamma=0.1
n_iter=100
rank=5
l0=1e-100
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
tag="test"
n_iter=1000
rank=5
lr=1e-1

OUTPUT="_out/"$INPUT"/"$input_type"/train_ratio_"$train_ratio"/rank_"$rank"/niter_"$n_iter"/tag_"$tag
python3 factorization/parafac/main.py    --input_tag $INPUT \
                            --input_type $input_type \
                            --out_dir $OUTPUT \
                            --entities $entities \
                            --value_column $value_column \
                            --train_ratio $train_ratio \
                            --rank $rank \
                            --n_iter $n_iter \
                            --lr $lr \
                            # 
fi