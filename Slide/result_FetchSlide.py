import numpy as np
import matplotlib.pyplot as plt
env_name = 'FetchSlide'
#
NoBC_score = np.load(f"/home/zhu_y@WMGDS.WMG.WARWICK.AC.UK/PycharmProjects/pythonProject/Results/{env_name}/Demo/TD3_HER_S5_score.npy")
BC_only_score = np.load(f"/home/zhu_y@WMGDS.WMG.WARWICK.AC.UK/PycharmProjects/pythonProject/Results/{env_name}/TD3_BC_only/RanNoise0.1/BC_S5_score.npy")
Qfilter = np.load(f"/home/zhu_y@WMGDS.WMG.WARWICK.AC.UK/PycharmProjects/pythonProject/Results/{env_name}/StdQfilter/RanNoise0.1/Qfilter_S5_score.npy")
x=np.arange(0, 5, 0.025)
plt.plot(x, NoBC_score[:10000:50], color='black', label='NoBC')
plt.plot(x, BC_only_score[:10000:50], color='green', label='BC_only')
plt.plot(x, Qfilter[:10000:50], color='purple', label='Qfilter')
# plt.plot(x, Qfilter_score_ensemble_mean[:10000:50], color='blue', label='Qfilter_ensemble')

plt.title('Scores')
plt.xlabel('Environment interactions (5e6)')
plt.ylabel('Score')
plt.legend()
# plt.savefig('/home/zhu_y@WMGDS.WMG.WARWICK.AC.UK/PycharmProjects/pythonProject/Figure/Slide/scores_BC_noNoise.png')
plt.show()
# # Plot the scores
NoBC_score = np.load(f"/home/zhu_y@WMGDS.WMG.WARWICK.AC.UK/PycharmProjects/pythonProject/Results/{env_name}/Demo/TD3_HER_S5_score.npy")
BC_only_score_s5 = np.load(f"/home/zhu_y@WMGDS.WMG.WARWICK.AC.UK/PycharmProjects/pythonProject/Results/{env_name}/TD3_BC_only/noNoise/BC_S5_score.npy")
BC_only_score_s1 = np.load(f"/home/zhu_y@WMGDS.WMG.WARWICK.AC.UK/PycharmProjects/pythonProject/Results/{env_name}/TD3_BC_only/noNoise/BC_S1_score.npy")
BC_only_score_s2 = np.load(f"/home/zhu_y@WMGDS.WMG.WARWICK.AC.UK/PycharmProjects/pythonProject/Results/{env_name}/TD3_BC_only/noNoise/BC_S2_score.npy")
BC_only_score_s3 = np.load(f"/home/zhu_y@WMGDS.WMG.WARWICK.AC.UK/PycharmProjects/pythonProject/Results/{env_name}/TD3_BC_only/noNoise/BC_S3_score.npy")
BC_only_score_mean = (BC_only_score_s1+BC_only_score_s2+BC_only_score_s3+BC_only_score_s5)/4

Qfilter_s1 = np.load(f"/home/zhu_y@WMGDS.WMG.WARWICK.AC.UK/PycharmProjects/pythonProject/Results/{env_name}/StdQfilter/noNoise/Qfilter_S1_score.npy")
Qfilter_s2 = np.load(f"/home/zhu_y@WMGDS.WMG.WARWICK.AC.UK/PycharmProjects/pythonProject/Results/{env_name}/StdQfilter/noNoise/Qfilter_S2_score.npy")
Qfilter_s3 = np.load(f"/home/zhu_y@WMGDS.WMG.WARWICK.AC.UK/PycharmProjects/pythonProject/Results/{env_name}/StdQfilter/noNoise/Qfilter_S3_score.npy")
Qfilter_s5 = np.load(f"/home/zhu_y@WMGDS.WMG.WARWICK.AC.UK/PycharmProjects/pythonProject/Results/{env_name}/StdQfilter/noNoise/Qfilter_S5_score.npy")
Qfilter_mean = (Qfilter_s1+Qfilter_s2+Qfilter_s3+Qfilter_s5)/4

Qfilter_score_ensemble_s1 = np.load(f"/home/zhu_y@WMGDS.WMG.WARWICK.AC.UK/PycharmProjects/pythonProject/Results/{env_name}/EnsQfilter/noNoise/First/EnsSize_10_S1_score.npy")
Qfilter_score_ensemble_s2 = np.load(f"/home/zhu_y@WMGDS.WMG.WARWICK.AC.UK/PycharmProjects/pythonProject/Results/{env_name}/EnsQfilter/noNoise/First/EnsSize_10_S2_score.npy")
Qfilter_score_ensemble_s3 = np.load(f"/home/zhu_y@WMGDS.WMG.WARWICK.AC.UK/PycharmProjects/pythonProject/Results/{env_name}/EnsQfilter/noNoise/First/EnsSize_10_S3_score.npy")
Qfilter_score_ensemble_s4 = np.load(f"/home/zhu_y@WMGDS.WMG.WARWICK.AC.UK/PycharmProjects/pythonProject/Results/{env_name}/EnsQfilter/noNoise/First/EnsSize_10_S4_score.npy")
Qfilter_score_ensemble_s5 = np.load(f"/home/zhu_y@WMGDS.WMG.WARWICK.AC.UK/PycharmProjects/pythonProject/Results/{env_name}/EnsQfilter/noNoise/First/EnsSize_10_S5_score.npy")
Qfilter_score_ensemble_mean = (Qfilter_score_ensemble_s1+Qfilter_score_ensemble_s2+Qfilter_score_ensemble_s3+Qfilter_score_ensemble_s4+Qfilter_score_ensemble_s5)/5

x=np.arange(0, 5, 0.025)
plt.plot(x, NoBC_score[:10000:50], color='black', label='NoBC')
plt.plot(x, BC_only_score_s1[:10000:50], color='green', alpha=0.2)
plt.plot(x, BC_only_score_s2[:10000:50], color='green', alpha=0.2)
plt.plot(x, BC_only_score_s3[:10000:50], color='green', alpha=0.2)
plt.plot(x, BC_only_score_s5[:10000:50], color='green', alpha=0.2)

plt.plot(x, Qfilter_s1[:10000:50], color='purple', alpha=0.2)
plt.plot(x, Qfilter_s2[:10000:50], color='purple', alpha=0.2)
plt.plot(x, Qfilter_s3[:10000:50], color='purple', alpha=0.2)
plt.plot(x, Qfilter_s5[:10000:50], color='purple', alpha=0.2)

plt.plot(x, Qfilter_score_ensemble_s1[:10000:50], color='blue', alpha=0.2)
plt.plot(x, Qfilter_score_ensemble_s2[:10000:50], color='blue', alpha=0.2)
plt.plot(x, Qfilter_score_ensemble_s3[:10000:50], color='blue', alpha=0.2)
plt.plot(x, Qfilter_score_ensemble_s4[:10000:50], color='blue', alpha=0.2)
plt.plot(x, Qfilter_score_ensemble_s5[:10000:50], color='blue', alpha=0.2)

plt.plot(x, BC_only_score_mean[:10000:50], color='green', label='BC_only')
plt.plot(x, Qfilter_mean[:10000:50], color='purple', label='Qfilter')
plt.plot(x, Qfilter_score_ensemble_mean[:10000:50], color='blue', label='Qfilter_ensemble')

plt.title('Scores')
plt.xlabel('Environment interactions (5e6)')
plt.ylabel('Score')
plt.legend()
plt.savefig('/home/zhu_y@WMGDS.WMG.WARWICK.AC.UK/PycharmProjects/pythonProject/Figure/Slide/scores_BC_noNoise.png')
plt.show()


first_score_s1 = np.load(f"/home/zhu_y@WMGDS.WMG.WARWICK.AC.UK/PycharmProjects/pythonProject/Results/{env_name}/EnsQfilter/noNoise/First/EnsSize_10_S1_score.npy")
first_score_s2 = np.load(f"/home/zhu_y@WMGDS.WMG.WARWICK.AC.UK/PycharmProjects/pythonProject/Results/{env_name}/EnsQfilter/noNoise/First/EnsSize_10_S2_score.npy")
first_score_s3 = np.load(f"/home/zhu_y@WMGDS.WMG.WARWICK.AC.UK/PycharmProjects/pythonProject/Results/{env_name}/EnsQfilter/noNoise/First/EnsSize_10_S3_score.npy")
first_score_s4 = np.load(f"/home/zhu_y@WMGDS.WMG.WARWICK.AC.UK/PycharmProjects/pythonProject/Results/{env_name}/EnsQfilter/noNoise/First/EnsSize_10_S4_score.npy")
first_score_s5 = np.load(f"/home/zhu_y@WMGDS.WMG.WARWICK.AC.UK/PycharmProjects/pythonProject/Results/{env_name}/EnsQfilter/noNoise/First/EnsSize_10_S5_score.npy")
first_score_mean = (first_score_s1+first_score_s2+first_score_s3+first_score_s4+first_score_s5)/5

# min_score = np.load(f"/home/zhu_y@WMGDS.WMG.WARWICK.AC.UK/PycharmProjects/pythonProject/Results/{env_name}/EnsQfilter/noNoise/Minimum/EnsSize_10_S5_score.npy")
lcb_score = np.load(f"/home/zhu_y@WMGDS.WMG.WARWICK.AC.UK/PycharmProjects/pythonProject/Results/{env_name}/EnsQfilter/noNoise/LCB/EnsSize_10_S5_score.npy")
mean_score_s1 = np.load(f"/home/zhu_y@WMGDS.WMG.WARWICK.AC.UK/PycharmProjects/pythonProject/Results/{env_name}/EnsQfilter/noNoise/Mean/EnsSize_10_S1_score.npy")
mean_score_s2 = np.load(f"/home/zhu_y@WMGDS.WMG.WARWICK.AC.UK/PycharmProjects/pythonProject/Results/{env_name}/EnsQfilter/noNoise/Mean/EnsSize_10_S2_score.npy")
mean_score_s3 = np.load(f"/home/zhu_y@WMGDS.WMG.WARWICK.AC.UK/PycharmProjects/pythonProject/Results/{env_name}/EnsQfilter/noNoise/Mean/EnsSize_10_S3_score.npy")
mean_score_s4 = np.load(f"/home/zhu_y@WMGDS.WMG.WARWICK.AC.UK/PycharmProjects/pythonProject/Results/{env_name}/EnsQfilter/noNoise/Mean/EnsSize_10_S4_score.npy")
mean_score_s5 = np.load(f"/home/zhu_y@WMGDS.WMG.WARWICK.AC.UK/PycharmProjects/pythonProject/Results/{env_name}/EnsQfilter/noNoise/Mean/EnsSize_10_S5_score.npy")
mean_score_mean = (mean_score_s1+mean_score_s2+mean_score_s3+mean_score_s4+mean_score_s5)/5
# newfirst_score = np.load(f"/home/zhu_y@WMGDS.WMG.WARWICK.AC.UK/PycharmProjects/pythonProject/Results/{env_name}/EnsQfilter/Noise0.2/NewFirst/NEW_EnsSize_10_S5_score.npy")

plt.plot(x, first_score_s1[:10000:50], color='blue', alpha=0.2)
plt.plot(x, first_score_s2[:10000:50], color='blue', alpha=0.2)
plt.plot(x, first_score_s3[:10000:50], color='blue', alpha=0.2)
plt.plot(x, first_score_s4[:10000:50], color='blue', alpha=0.2)
plt.plot(x, first_score_s5[:10000:50], color='blue', alpha=0.2)

# plt.plot(x, min_score[:10000:100], color='blue', label='Min')
plt.plot(x, mean_score_s1[:10000:50], color='red', alpha=0.2)
plt.plot(x, mean_score_s2[:10000:50], color='red', alpha=0.2)
plt.plot(x, mean_score_s3[:10000:50], color='red', alpha=0.2)
plt.plot(x, mean_score_s4[:10000:50], color='red', alpha=0.2)
plt.plot(x, mean_score_s5[:10000:50], color='red', alpha=0.2)

plt.plot(x, lcb_score[:10000:50], color='green', label='LCB')
plt.plot(x, first_score_mean[:10000:50], color='blue', label='Qfilter')
plt.plot(x, mean_score_mean[:10000:50], color='red', label='Mean')
plt.title('Scores')
plt.xlabel('Environment interactions')
plt.ylabel('Score')
plt.legend()
plt.savefig('/home/zhu_y@WMGDS.WMG.WARWICK.AC.UK/PycharmProjects/pythonProject/Figure/Slide/scores_noNoise.png')
plt.show()

x=np.arange(0, 5, 0.05)
first_success = np.load(f"/home/zhu_y@WMGDS.WMG.WARWICK.AC.UK/PycharmProjects/pythonProject/Results/{env_name}/EnsQfilter/noNoise/First/EnsSize_10_S5_success.npy")
mean_success = np.load(f"/home/zhu_y@WMGDS.WMG.WARWICK.AC.UK/PycharmProjects/pythonProject/Results/{env_name}/EnsQfilter/noNoise/Mean/EnsSize_10_S5_success.npy")
lcb_success = np.load(f"/home/zhu_y@WMGDS.WMG.WARWICK.AC.UK/PycharmProjects/pythonProject/Results/{env_name}/EnsQfilter/noNoise/LCB/EnsSize_10_S5_success.npy")
# modifiedlcb_success = np.load(f"/home/zhu_y@WMGDS.WMG.WARWICK.AC.UK/PycharmProjects/pythonProject/Results/{env_name}/EnsQfilter/Noise0.2/ModifiedLCB/EnsSize_10_S5_success.npy")
# newfirst_score = np.load(f"/home/zhu_y@WMGDS.WMG.WARWICK.AC.UK/PycharmProjects/pythonProject/Results/{env_name}/EnsQfilter/Noise0.2/NewFirst/NEW_EnsSize_10_S5_success.npy")
plt.plot(x, first_success[:10000:100], color='blue', label='First')
plt.plot(x, mean_success[:10000:100], color='red', label='Mean')
plt.plot(x, lcb_success[:10000:100], color='green', label='LCB')
# plt.plot(x, modifiedlcb_success[:10000:100], color='red', label='ModifiedLCB')
plt.title('Success Rate')
plt.xlabel('Environment interactions')
plt.ylabel('Success Rate')
plt.legend()
plt.savefig('/home/zhu_y@WMGDS.WMG.WARWICK.AC.UK/PycharmProjects/pythonProject/Figure/Slide/success_noNoise.png')
plt.show()
#

x=np.arange(0, 5, 0.025)
first_demoaccept = np.load(f"/home/zhu_y@WMGDS.WMG.WARWICK.AC.UK/PycharmProjects/pythonProject/Results/{env_name}/EnsQfilter/noNoise/First/EnsSize_10_S5_demoaccept.npy")
mean_demoaccept = np.load(f"/home/zhu_y@WMGDS.WMG.WARWICK.AC.UK/PycharmProjects/pythonProject/Results/{env_name}/EnsQfilter/noNoise/Mean/EnsSize_10_S5_demoaccept.npy")
lcb_demoaccept = np.load(f"/home/zhu_y@WMGDS.WMG.WARWICK.AC.UK/PycharmProjects/pythonProject/Results/{env_name}/EnsQfilter/noNoise/LCB/EnsSize_10_S5_demoaccept.npy")
# modifiedlcb_demoaccept = np.load(f"/home/zhu_y@WMGDS.WMG.WARWICK.AC.UK/PycharmProjects/pythonProject/Results/{env_name}/EnsQfilter/Noise0.2/ModifiedLCB/EnsSize_10_S5_demoaccept.npy")
# newfirst_demoaccept = np.load(f"/home/zhu_y@WMGDS.WMG.WARWICK.AC.UK/PycharmProjects/pythonProject/Results/{env_name}/EnsQfilter/Noise0.2/NewFirst/NEW_EnsSize_10_S5_demoaccept.npy")
plt.plot(x, first_demoaccept[:10000:100], color='blue', label='First')
plt.plot(x, mean_demoaccept[:10000:100], color='red', label='Mean')
plt.plot(x, lcb_demoaccept[:10000:100], color='green', label='LCB')
# plt.plot(x, modifiedlcb_demoaccept[:10000:100], color='red', label='ModifiedLCB')
plt.title('Accpetance Rate of demonstrations')
plt.xlabel('Environment interactions')
plt.ylabel('Accpetance Rate')
plt.legend()
plt.savefig('/home/zhu_y@WMGDS.WMG.WARWICK.AC.UK/PycharmProjects/pythonProject/Figure/Slide/demoaccepts_noNoise.png')
plt.show()


