#!/bin/sh

allocation_name='loni_ethicalrl'
job_name='ethical_rl'
queue_name='gpu'
nodes='1'
ppn='20'
walltime='01:00:00'
email=''
output=''
error=''
singularity_image='/home/admin/singularity/tf.2.1.1.gpu.simg'
requirements_file="$PWD/requirements.txt"
driver_script="$PWD/main.py"
test_name=''
test_args=''
number_of_experiments=1

print_usage() {
  printf "Usage: ..."
}

while getopts 'c:t:a:j:q:n:p:w:e:o:' flag; do
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
    o) output="${OPTARG}/stdout" 
       error="${OPTARG}/stderr" 
        ;;
    *) print_usage
       exit 1 ;;
  esac
done

# build shell script
shell_script='#!/bin/sh \n'
while read -r line
do 
  echo "$line"
  shell_script="${shell_script}python -m pip install ${line} \n"
done < "$requirements_file"
shell_script="${shell_script}python -m pip install --no-deps git+https://github.com/maximecb/gym-minigrid.git \n"
shell_script="${shell_script}python -m pip install --no-deps git+https://github.com/arie-glazier/ethical_rl.git \n"
shell_script="${shell_script}python ${driver_script} --test_name ${test_name} ${test_args}"
printf "${shell_script}" > ${job_name}.sh
chmod 777 ${job_name}.sh

# set args
pbs_file='#!/bin/sh \n'
pbs_file="${pbs_file}#PBS -A ${allocation_name} \n"
pbs_file="${pbs_file}#PBS -q ${queue_name} \n"
pbs_file="${pbs_file}#PBS -l nodes=${nodes}:ppn=${ppn} \n"
pbs_file="${pbs_file}#PBS -l walltime=${walltime} \n"
if [[ $email ]]
then
  echo "email: *${email}*"
  pbs_file="${pbs_file}#PBS -m e \n"
  pbs_file="${pbs_file}#PBS -M ${email} \n"
fi
pbs_file="${pbs_file}#PBS -o ${output} \n"
pbs_file="${pbs_file}#PBS -e ${error} \n"

# navigate
pbs_file="${pbs_file}cd \$PBS_O_WORKDIR \n"

# execute
pbs_file="${pbs_file}singularity exec --nv -B /work/$USER ${singularity_image} ${PWD}/${job_name}.sh \n"

printf "${pbs_file}" > ${job_name}.pbs
chmod 777 ${job_name}.pbs

for ((i=1;i<=number_of_experiments;i++)); do
  echo "qsub ${job_name}.pbs" | bash
done