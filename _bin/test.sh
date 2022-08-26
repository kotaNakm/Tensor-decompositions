cd `dirname $0`
cd ..


INPUT="gowalla"
OUTPUT="_out/test"
input_type="AGH_experiment1"

entities="userid/placeid/yr_month"
value_column="checkins"

if true;then
python3 _src/AGH/main.py    --input_tag $INPUT \
                            --input_type $input_type \
                            --out_dir $OUTPUT \
                            --entities $entities \
                            --value_column $value_column \
                            --initial_gamma 3 \
                            --rank 5 \
                            --n_iter 100 \
                            # 
fi