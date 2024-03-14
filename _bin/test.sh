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

<<<<<<< HEAD
OUTPUT="_out/"$tag"/train_ratio_"$train_ratio"/rank_"$rank"/gamma_"$gamma"/l0_"$l0"/niter_"$n_iter
=======
OUTPUT="_out/"$tag"/train_ratio_"$train_ratio"/rank_"$rank"/gamma_"$gamma"/niter_"$n_iter

>>>>>>> refs/remotes/origin/main
if true;then
python3 _src/AGH/main.py    --input_tag $INPUT \
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