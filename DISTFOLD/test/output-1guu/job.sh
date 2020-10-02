#!/bin/bash                                       
echo "starting cns.."                           
touch iam.running                                 
# CNS-CONFIGURATION                               
source /data/ieee-icmla-2019/DISTFOLD-v0.1/cns_solve_1.3/cns_solve_env.sh                
export KMP_AFFINITY=none                          
export CNS_CUSTOMMODULE=/data/ieee-icmla-2019/DISTFOLD-v0.1/test/output-1guu/                 
/data/ieee-icmla-2019/DISTFOLD-v0.1/cns_solve_1.3/intel-x86_64bit-linux/bin/cns_solve < dgsa.inp > dgsa.log 
if [ -f "1guu_20.pdb" ]; then           
   rm iam.running                                 
   echo "trial structures written."             
   rm *embed*                                     
   exit                                           
fi                                                
if [ -f "1guua_20.pdb" ]; then 
   rm iam.running                                 
   echo "accepted structures written."          
   rm *embed*                                     
   exit                                           
fi                                                
tail -n 30 dgsa.log                              
echo "ERROR! Final structures not found!"       
echo "CNS FAILED!"                              
mv iam.running iam.failed                         
