import numpy as np
import matplotlib.pyplot as plt
env_name = 'FetchPush'
x=np.arange(0, 2, 0.01)

# # Plot the scores
# NoBC_score = np.load(f"/home/zhu_y@WMGDS.WMG.WARWICK.AC.UK/PycharmProjects/pythonProject/Results/{env_name}/Demo/TD3_HER_S5_score2.npy")
# BC_only_score = np.load(f"/home/zhu_y@WMGDS.WMG.WARWICK.AC.UK/PycharmProjects/pythonProject/Results/{env_name}/TD3_BC_only/RandGausNoise/0.5+1BC_S5_score.npy")
# Qfilter_score = np.load(f"/home/zhu_y@WMGDS.WMG.WARWICK.AC.UK/PycharmProjects/pythonProject/Results/{env_name}/StdQfilter/RandGausNoise/0.5+1Qfilter_S5_score.npy")
# plt.plot(x, NoBC_score[:4000:20], color='black', label='NoBC')
# plt.plot(x, BC_only_score[:4000:20], color='green', label='BC_only')
# plt.plot(x, Qfilter_score[:4000:20], color='purple', label='Qfilter')
# plt.title('Scores')
# plt.xlabel('Environment interactions (2e6)')
# plt.ylabel('Score')
# plt.legend()
# # plt.savefig('/home/zhu_y@WMGDS.WMG.WARWICK.AC.UK/PycharmProjects/pythonProject/Figure/Push/Noise0.5+1/scores_BC.png')
# plt.show()

demoAccept_first = np.load(f"/home/zhu_y@WMGDS.WMG.WARWICK.AC.UK/PycharmProjects/pythonProject/Results/{env_name}/EnsQfilter/RandGausNoise/First/EnsSize_10_S5_demoaccept.npy")
demoAccept_first_pretrain = np.load(f"/home/zhu_y@WMGDS.WMG.WARWICK.AC.UK/PycharmProjects/pythonProject/Results/{env_name}/EnsQfilter/Pretrain/First_EnsSize_10_S5_demoaccept.npy")
demoAccept_mean = np.load(f"/home/zhu_y@WMGDS.WMG.WARWICK.AC.UK/PycharmProjects/pythonProject/Results/{env_name}/EnsQfilter/RandGausNoise/Mean/EnsSize_10_S5_demoaccept.npy")
demoAccept_mean_pretrain = np.load(f"/home/zhu_y@WMGDS.WMG.WARWICK.AC.UK/PycharmProjects/pythonProject/Results/{env_name}/EnsQfilter/Pretrain/Mean_EnsSize_10_S5_demoaccept.npy")
demoAccept_lcb = np.load(f"/home/zhu_y@WMGDS.WMG.WARWICK.AC.UK/PycharmProjects/pythonProject/Results/{env_name}/EnsQfilter/RandGausNoise/LCB/EnsSize_10_S5_demoaccept.npy")
demoAccept_lcb_pretrain = np.load(f"/home/zhu_y@WMGDS.WMG.WARWICK.AC.UK/PycharmProjects/pythonProject/Results/{env_name}/EnsQfilter/Pretrain/LCB_EnsSize_10_S5_demoaccept.npy")
plt.plot(x, demoAccept_first[1:4000:20], color='black', label='Qfilter')
plt.plot(x, demoAccept_first_pretrain[1::20], color='grey', label='Qfilter_pretrain')
plt.plot(x, demoAccept_mean[1:4000:20], color='blue', label='Mean')
plt.plot(x, demoAccept_mean_pretrain[1::20], color='green', label='Mean_pretrain')
plt.plot(x, demoAccept_lcb[1:4000:20], color='red', label='LCB')
plt.plot(x, demoAccept_lcb_pretrain[1::20], color='orange', label='LCB_pretrain')
plt.title('Acceptance rate of demonstrations')
plt.xlabel('Environment interactions (2e6)')
plt.ylabel('Acceptance rate')
plt.legend()
plt.savefig('/home/zhu_y@WMGDS.WMG.WARWICK.AC.UK/PycharmProjects/pythonProject/Figure/Push/Noise0.5+1/demoaccept_pretrain.png')
plt.show()


# success_first = np.load(f"/home/zhu_y@WMGDS.WMG.WARWICK.AC.UK/PycharmProjects/pythonProject/Results/{env_name}/EnsQfilter/RandGausNoise/First/0.5+1EnsSize_10_S5_success.npy")
# success_first_pretrain = np.load(f"/home/zhu_y@WMGDS.WMG.WARWICK.AC.UK/PycharmProjects/pythonProject/Results/{env_name}/EnsQfilter/Pretrain/First_EnsSize_10_S5_success.npy")
# success_lcb = np.load(f"/home/zhu_y@WMGDS.WMG.WARWICK.AC.UK/PycharmProjects/pythonProject/Results/{env_name}/EnsQfilter/RandGausNoise/LCB/0.5+1EnsSize_10_S5_success.npy")
# success_lcb_pretrain = np.load(f"/home/zhu_y@WMGDS.WMG.WARWICK.AC.UK/PycharmProjects/pythonProject/Results/{env_name}/EnsQfilter/Pretrain/LCB_EnsSize_10_S5_success.npy")
# plt.plot(x, success_first[1::20], color='blue', label='Qfilter')
# plt.plot(x, success_first_pretrain[1::20], color='green', label='Qfilter_pretrain')
# plt.plot(x, success_lcb[1::20], color='red', label='LCB')
# plt.plot(x, success_lcb_pretrain[1::20], color='orange', label='LCB_pretrain')
# plt.title('Success rate of demonstrations')
# plt.xlabel('Environment interactions (2e6)')
# plt.ylabel('Success rate')
# plt.legend()
# # plt.savefig('/home/zhu_y@WMGDS.WMG.WARWICK.AC.UK/PycharmProjects/pythonProject/Figure/Push/Noise0.5+1/success.png')
# plt.show()

score_first = np.load(f"/home/zhu_y@WMGDS.WMG.WARWICK.AC.UK/PycharmProjects/pythonProject/Results/{env_name}/EnsQfilter/RandGausNoise/First/EnsSize_10_S5_score.npy")
score_first_pretrain = np.load(f"/home/zhu_y@WMGDS.WMG.WARWICK.AC.UK/PycharmProjects/pythonProject/Results/{env_name}/EnsQfilter/Pretrain/First_EnsSize_10_S5_score.npy")
score_mean_pretrain_fixQ = np.load(f"/home/zhu_y@WMGDS.WMG.WARWICK.AC.UK/PycharmProjects/pythonProject/Results/{env_name}/EnsQfilter/Pretrain/FixedQ/Steps1e4/Mean_EnsSize_10_S5_score.npy")

score_mean = np.load(f"/home/zhu_y@WMGDS.WMG.WARWICK.AC.UK/PycharmProjects/pythonProject/Results/{env_name}/EnsQfilter/RandGausNoise/Mean/EnsSize_10_S5_score.npy")
score_mean_pretrain = np.load(f"/home/zhu_y@WMGDS.WMG.WARWICK.AC.UK/PycharmProjects/pythonProject/Results/{env_name}/EnsQfilter/Pretrain/Steps1e4/Mean_EnsSize_10_S5_score.npy")
score_lcb = np.load(f"/home/zhu_y@WMGDS.WMG.WARWICK.AC.UK/PycharmProjects/pythonProject/Results/{env_name}/EnsQfilter/RandGausNoise/LCB/EnsSize_10_S5_score.npy")
score_lcb_pretrain = np.load(f"/home/zhu_y@WMGDS.WMG.WARWICK.AC.UK/PycharmProjects/pythonProject/Results/{env_name}/EnsQfilter/Pretrain/LCB_EnsSize_10_S5_score.npy")

plt.plot(x, score_first[1:4000:20], color='black', label='Qfilter')
# plt.plot(x, score_first_pretrain[1::20], color='grey', label='Qfilter_pretrain')
plt.plot(x, score_mean_pretrain_fixQ[1::20], color='red', label='Mean_pretrain_fixQ')
plt.plot(x, score_mean[1:4000:20], color='blue', label='Mean')
plt.plot(x, score_mean_pretrain[1::20], color='green', label='Mean_pretrain')
# plt.plot(x, score_lcb[1:4000:20], color='red', label='LCB')
# plt.plot(x, score_lcb_pretrain[1::20], color='orange', label='LCB_pretrain')
plt.title('Scores')
plt.xlabel('Environment interactions (2e6)')
plt.ylabel('Score')
plt.legend()
plt.savefig('/home/zhu_y@WMGDS.WMG.WARWICK.AC.UK/PycharmProjects/pythonProject/Figure/Push/Noise0.5+1/scores_pretrain_fixQ_steps1e4.png')
plt.show()


# score_first = np.load(f"/home/zhu_y@WMGDS.WMG.WARWICK.AC.UK/PycharmProjects/pythonProject/Results/{env_name}/EnsQfilter/RandGausNoise/First/EnsSize_10_S5_score.npy")
# score_first_pretrain = np.load(f"/home/zhu_y@WMGDS.WMG.WARWICK.AC.UK/PycharmProjects/pythonProject/Results/{env_name}/EnsQfilter/Pretrain/First_EnsSize_10_S5_score.npy")
# score_mean = np.load(f"/home/zhu_y@WMGDS.WMG.WARWICK.AC.UK/PycharmProjects/pythonProject/Results/{env_name}/EnsQfilter/RandGausNoise/Mean/EnsSize_10_S5_score.npy")
# score_mean_pretrain = np.load(f"/home/zhu_y@WMGDS.WMG.WARWICK.AC.UK/PycharmProjects/pythonProject/Results/{env_name}/EnsQfilter/Pretrain/Mean_EnsSize_10_S5_score.npy")
# score_lcb = np.load(f"/home/zhu_y@WMGDS.WMG.WARWICK.AC.UK/PycharmProjects/pythonProject/Results/{env_name}/EnsQfilter/RandGausNoise/LCB/EnsSize_10_S5_score.npy")
# score_lcb_pretrain = np.load(f"/home/zhu_y@WMGDS.WMG.WARWICK.AC.UK/PycharmProjects/pythonProject/Results/{env_name}/EnsQfilter/Pretrain/LCB_EnsSize_10_S5_score.npy")
#
# plt.plot(x, score_first[1:4000:20], color='black', label='Qfilter')
# plt.plot(x, score_first_pretrain[1::20], color='grey', label='Qfilter_pretrain')
# plt.plot(x, score_mean[1:4000:20], color='blue', label='Mean')
# plt.plot(x, score_mean_pretrain[1::20], color='green', label='Mean_pretrain')
# plt.plot(x, score_lcb[1:4000:20], color='red', label='LCB')
# plt.plot(x, score_lcb_pretrain[1::20], color='orange', label='LCB_pretrain')
# plt.title('Scores')
# plt.xlabel('Environment interactions (2e6)')
# plt.ylabel('Score')
# plt.legend()
# plt.savefig('/home/zhu_y@WMGDS.WMG.WARWICK.AC.UK/PycharmProjects/pythonProject/Figure/Push/Noise0.5+1/scores_pretrain.png')
# plt.show()
