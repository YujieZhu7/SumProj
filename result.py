import numpy as np
import matplotlib.pyplot as plt
env_name = 'FetchPush'

# Plot the scores
NoBC_score = np.load(f"/home/zhu_y@WMGDS.WMG.WARWICK.AC.UK/PycharmProjects/pythonProject/Results/{env_name}/TD3_HER_S5_score.npy")
# Qfilter_score = np.load(f"/home/zhu_y@WMGDS.WMG.WARWICK.AC.UK/PycharmProjects/pythonProject/Results/{env_name}/StdQfilter/Qfilter_S5_score.npy")
Qfilter_score2 = np.load(f"/home/zhu_y@WMGDS.WMG.WARWICK.AC.UK/PycharmProjects/pythonProject/Results/{env_name}/StdQfilter/Qfilter_S5_score2.npy")
BC_only_score = np.load(f"/home/zhu_y@WMGDS.WMG.WARWICK.AC.UK/PycharmProjects/pythonProject/Results/{env_name}/TD3_BC_only/BC_S5_score.npy")

x1=list(range(len(NoBC_score)))
x2=list(range(len(Qfilter_score2)))
plt.plot(x1[::50],NoBC_score[::50], color='black', label='NoBC')
# plt.plot(x2,Qfilter_score, color='blue', label='Qfilter')
plt.plot(x2[::50],Qfilter_score2[::50], color='red', label='Qfilter')
plt.plot(x2[::50],BC_only_score[::50], color='green', label='BC_only')

plt.title('Scores')
plt.xlabel('Environment interactions')
plt.ylabel('Score')
plt.legend()
plt.savefig('/home/zhu_y@WMGDS.WMG.WARWICK.AC.UK/PycharmProjects/pythonProject/Figure/scores.png')
plt.show()

# Plot the success rate
NoBC_success = np.load(f"/home/zhu_y@WMGDS.WMG.WARWICK.AC.UK/PycharmProjects/pythonProject/Results/{env_name}/TD3_HER_S5_success.npy")
# Qfilter_success = np.load(f"/home/zhu_y@WMGDS.WMG.WARWICK.AC.UK/PycharmProjects/pythonProject/Results/{env_name}/StdQfilter/Qfilter_S5_sucess.npy")
Qfilter_success2 = np.load(f"/home/zhu_y@WMGDS.WMG.WARWICK.AC.UK/PycharmProjects/pythonProject/Results/{env_name}/StdQfilter/Qfilter_S5_sucess2.npy")
BC_only_success = np.load(f"/home/zhu_y@WMGDS.WMG.WARWICK.AC.UK/PycharmProjects/pythonProject/Results/{env_name}/TD3_BC_only/BC_S5_sucess.npy")

x4=np.arange(0, 2, 0.005)
plt.plot(x4,NoBC_success[:4000:10], color='black', label='NoBC')
# plt.plot(range(4000),Qfilter_success[:4000], color='blue', label='Qfilter')
plt.plot(x4,Qfilter_success2[:4000:10], color='red', label='Qfilter')
plt.plot(x4,BC_only_success[:4000:10], color='green', label='BC_only')

plt.title('Success rate of first 2e6 steps')
plt.xlabel('Environment interactions (1e6)')
plt.ylabel('Success rate')
plt.legend()
plt.savefig('/home/zhu_y@WMGDS.WMG.WARWICK.AC.UK/PycharmProjects/pythonProject/Figure/success_rates_2e6.png')
plt.show()

# Plot the success rate (zoom in)
NoBC_success = np.load(f"/home/zhu_y@WMGDS.WMG.WARWICK.AC.UK/PycharmProjects/pythonProject/Results/{env_name}/TD3_HER_S5_success.npy")
# Qfilter_success = np.load(f"/home/zhu_y@WMGDS.WMG.WARWICK.AC.UK/PycharmProjects/pythonProject/Results/{env_name}/StdQfilter/Qfilter_S5_sucess.npy")
Qfilter_success2 = np.load(f"/home/zhu_y@WMGDS.WMG.WARWICK.AC.UK/PycharmProjects/pythonProject/Results/{env_name}/StdQfilter/Qfilter_S5_sucess2.npy")
BC_only_success = np.load(f"/home/zhu_y@WMGDS.WMG.WARWICK.AC.UK/PycharmProjects/pythonProject/Results/{env_name}/TD3_BC_only/BC_S5_sucess.npy")

x3=np.arange(0,0.5, 0.0025)
plt.plot(x3,NoBC_success[:1000:5], color='black', label='NoBC')
# plt.plot(range(1000),Qfilter_success[:1000], color='blue', label='Qfilter')
plt.plot(x3,Qfilter_success2[:1000:5], color='red', label='Qfilter')
plt.plot(x3,BC_only_success[:1000:5], color='green', label='BC_only')

plt.title('Success rate of first 5e5 steps')
plt.xlabel('Environment interactions (1e6)')
plt.ylabel('Success rate')
plt.legend()
plt.savefig('/home/zhu_y@WMGDS.WMG.WARWICK.AC.UK/PycharmProjects/pythonProject/Figure/success_rates_5e5.png')
plt.show()

x5=np.arange(0, 4, 0.025)
demoAccept2 = np.load(f"/home/zhu_y@WMGDS.WMG.WARWICK.AC.UK/PycharmProjects/pythonProject/Results/{env_name}/StdQfilter/Qfilter_S5_demoaccept2.npy")
demoAccept = np.load(f"/home/zhu_y@WMGDS.WMG.WARWICK.AC.UK/PycharmProjects/pythonProject/Results/{env_name}/EnsQfilter/First/First_EnsSize_2_S5_demoaccept.npy")
plt.plot(x5, demoAccept[1::50], color='blue', label='Qfilter')
plt.plot(x5, demoAccept2[1::50], color='red', label='Qfilter2')
plt.title('Acceptance rate of demonstrations')
plt.xlabel('Environment interactions (1e6)')
plt.ylabel('Acceptance rate')
plt.legend()
plt.savefig('/home/zhu_y@WMGDS.WMG.WARWICK.AC.UK/PycharmProjects/pythonProject/Figure/acceptdemos.png')
plt.show()