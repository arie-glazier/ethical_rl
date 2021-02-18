#!/bin/sh

allocation_name='loni_ethicalrl'
job_name='ethical_rl'
queue_name='gpu'
nodes='1'
ppn='1'
walltime='02:00:00'
email=''
output='slurm-%%j.out-%%N'
error='slurm-%%j.err-%%N'
singularity_image='/home/admin/singularity/tf.2.1.1.gpu.simg'
requirements_file="$PWD/requirements.txt"
driver_script="$PWD/main.py"
test_name=''
experiment_name=''
test_args=''
number_of_experiments=1

print_usage() {
  printf "Usage: ..."
}

while getopts 'c:t:a:j:q:n:p:w:e:o:x:' flag; do
  echo $flag
  echo ${OPTARG}
  case "${flag}" in
    c) number_of_experiments="${OPTARG}" ;;
    e) email="${OPTARG}" ;;
    j) job_name="${OPTARG}" ;;
    q) queue_name="${OPTARG}" ;;
    n) nodes="${OPTARG}" ;;
    p) ppn="${OPTARG}" ;;
    w) walltime="${OPTARG}" ;;
    s) singularity_image="${OPTARG}" ;;
    t) test_name="${OPTARG}" ;;
    a) test_args="${OPTARG}" ;;
    x) experiment_name="${OPTARG}" ;;
    o) output="${OPTARG}/stdout" 
       error="${OPTARG}/stderr" 
        ;;
    *) print_usage
       exit 1 ;;
  esac
done

# set work_dir
WORK_DIR="/work/${USER}/${experiment_name}"
if [[ ! -e $WORK_DIR ]]
then
  mkdir -p $WORK_DIR
fi

# build shell script
shell_script='#!/bin/sh \n'
while read -r line
do 
  echo "$line"
  shell_script="${shell_script}python -m pip install ${line} \n"
done < "$requirements_file"
shell_script="${shell_script}python -m pip install --no-deps git+https://github.com/maximecb/gym-minigrid.git \n"
shell_script="${shell_script}python -m pip install --no-deps git+https://github.com/arie-glazier/ethical_rl.git \n"
shell_script="${shell_script}python -u ${driver_script} --test_name ${test_name} ${test_args}"
if [[ $experiment_name ]]
then
  shell_script="${shell_script} --experiment_group_name ${experiment_name}"
fi

JOB_FILE=$WORK_DIR/${job_name}_job.sh
printf "${shell_script}" > $JOB_FILE
chmod 777 $JOB_FILE

# set args
pbs_file='#!/bin/sh \n'
pbs_file="${pbs_file}#SBATCH -A ${allocation_name} \n"
pbs_file="${pbs_file}#SBATCH -p ${queue_name} \n"
pbs_file="${pbs_file}#SBATCH -N ${nodes} \n"
pbs_file="${pbs_file}#SBATCH -n ${ppn} \n"
pbs_file="${pbs_file}#SBATCH -t ${walltime} \n"
if [[ $email ]]
then
  echo "email: *${email}*"
  pbs_file="${pbs_file}#SBATCH --mail-user ${email} \n"
  pbs_file="${pbs_file}#SBATCH --mail-type END \n"
fi
pbs_file="${pbs_file}#SBATCH -o ${output} \n"
pbs_file="${pbs_file}#SBATCH -e ${error} \n"

# execute
pbs_file="${pbs_file}singularity exec --nv -B /work/$USER ${singularity_image} ${JOB_FILE} \n"

SBATCH_FILE=$WORK_DIR/${job_name}_sbatch.sh
printf "${pbs_file}" > $SBATCH_FILE
chmod 777 $SBATCH_FILE

cd $WORK_DIR
for ((i=1;i<=number_of_experiments;i++)); do
  echo "sbatch $SBATCH_FILE" | bash
done