import numpy as np
import matplotlib.pyplot as plt
env_name = 'FetchSlide'
#
# # Plot the scores
NoBC_score = np.load(f"/home/zhu_y@WMGDS.WMG.WARWICK.AC.UK/PycharmProjects/pythonProject/Results/{env_name}/Demo/TD3_HER_S5_success.npy")
# BC_only_score = np.load(f"/home/zhu_y@WMGDS.WMG.WARWICK.AC.UK/PycharmProjects/pythonProject/Results/{env_name}/TD3_BC_only/Noise0.5/BC_S5_score.npy")
Qfilter_score_min = np.load(f"/home/zhu_y@WMGDS.WMG.WARWICK.AC.UK/PycharmProjects/pythonProject/Results/{env_name}/EnsQfilter/Noise0.2/Minimum/EnsSize_10_S5_success.npy")
Qfilter_score_lcb = np.load(f"/home/zhu_y@WMGDS.WMG.WARWICK.AC.UK/PycharmProjects/pythonProject/Results/{env_name}/EnsQfilter/Noise0.2/LCB/EnsSize_10_S5_success.npy")
#
x=np.arange(0, 2, 0.01)
plt.plot(NoBC_score, color='black', label='NoBC')
# plt.plot(NoBC_score[::100], color='blue', label='No_BC')
plt.plot(Qfilter_score_min, color='red', label='Qfilter_min')
plt.plot(Qfilter_score_lcb, color='green', label='Qfilter_LCB')

plt.title('Scores')
plt.xlabel('Environment interactions')
plt.ylabel('Score')
plt.legend()
# plt.savefig('/home/zhu_y@WMGDS.WMG.WARWICK.AC.UK/PycharmProjects/pythonProject/Figure/scores_BC_noise0.5.png')
plt.show()

# env_name = 'FetchPush'
# var=0.5
# ensemble_size=10
# seed=5
# Qfilter=np.load(f"/home/zhu_y@WMGDS.WMG.WARWICK.AC.UK/PycharmProjects/pythonProject/Results/{env_name}/EnsQfilter/Noise{var}/First/EnsSize_{ensemble_size}_S{seed}_demoaccept.npy")
# Qfilter_pretrain = np.load(f"/home/zhu_y@WMGDS.WMG.WARWICK.AC.UK/PycharmProjects/pythonProject/Results/{env_name}/EnsQfilter/Pretrain/Noise{var}/First/EnsSize_{ensemble_size}_S{seed}_demoaccept.npy")
# LCB=np.load(f"/home/zhu_y@WMGDS.WMG.WARWICK.AC.UK/PycharmProjects/pythonProject/Results/{env_name}/EnsQfilter/Noise{var}/LCB/EnsSize_{ensemble_size}_S{seed}_demoaccept.npy")
# LCB_pretrain = np.load(f"/home/zhu_y@WMGDS.WMG.WARWICK.AC.UK/PycharmProjects/pythonProject/Results/{env_name}/EnsQfilter/Pretrain/Noise{var}/LCB/EnsSize_{ensemble_size}_S{seed}_demoaccept.npy")
# plt.plot(Qfilter[::10], color='black', label='Qfilter')
# plt.plot(Qfilter_pretrain[::10], color='blue', label='Qfilter_pretrain')
# plt.plot(LCB[::10], color='red', label='LCB')
# plt.plot(LCB_pretrain[::10], color='green', label='LCB_pretrain')
# plt.title('Acceptance rate of demonstrations')
# plt.xlabel('Environment interactions (2e6)')
# plt.ylabel('Acceptance rate')
# plt.legend()
# plt.savefig('/home/zhu_y@WMGDS.WMG.WARWICK.AC.UK/PycharmProjects/pythonProject/Figure/Push/demoaccept_ensize10_var0.5.png')
# plt.show()

