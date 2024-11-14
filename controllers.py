from func.config import *
import func.model_response as model_response
import func.cnn as cnn
import func.utils as utils
from scipy.optimize import minimize
import tensorflow.keras as keras
import numpy as np
import matplotlib.pyplot as plt


################################################################
################################################################
################################################################
################################################################
class Controllers():

	def __init__(self, scaler_x, scaler_y, model, input_DB, input_SN, r_CO, r_AP, sensitivities):
		self.scaler_x = scaler_x
		self.scaler_y = scaler_y
		self.model = model

		self.input_DB = input_DB
		self.input_SN = input_SN
		self.r_CO = r_CO
		self.r_AP = r_AP

		self.sensitivities = sensitivities



	#=========================================================#
	def math_model(self, input_DB, input_SN):
		"""	
		current time step response
		"""	

		(a1,b1,a2,b2) = self.sensitivities

		CO1_linear, CO1_nonlinear = model_response.response(u=input_DB, drug_response='DB-CO')
		CO2_linear, CO2_nonlinear = model_response.response(u=input_SN, drug_response='SN-CO')

		AP1_linear, AP1_nonlinear = model_response.response(u=input_SN, drug_response='SN-AP')
		AP2_linear, AP2_nonlinear = model_response.response(u=input_DB, drug_response='DB-AP')

		CO = a1*np.array(CO1_nonlinear) + a2*np.array(CO2_nonlinear)
		AP = b1*np.array(AP1_nonlinear) + b2*np.array(AP2_nonlinear)

		return CO[-1], AP[-1]




	#=========================================================#
	def model_predict_future(self, input_list):
	    """
	    Neural Network Model to predict (future time-step)
	    Input: list of 6 elements 
	    """
	    arr = np.array(input_list).reshape(1, 6)
	    arr_norm = self.scaler_x.transform(arr)
	    arr_test = arr_norm.reshape(-1, 3, 2, 1)
	    pred = self.model.predict(arr_test)
	    pred_convert = self.scaler_y.inverse_transform(pred)

	    return pred_convert[0,0], pred_convert[0,1]


	#=========================================================#
	def calculate_loss(self, x):
	    """
	    Calculate Loss Function
	    """
	    DB_current = x[0]
	    SN_current = x[1]

	    DB_minus1 = self.input_DB[-1]
	    SN_minus1 = self.input_SN[-1]

	    output_CO_current, output_AP_current = self.math_model(self.input_DB + [DB_current],
	                                                    		self.input_SN + [SN_current])

	    ## predict output_cnn[step+1]
	    list_input0 = [DB_minus1, DB_current, 
	                    SN_minus1, SN_current, 
	                    output_CO_current, output_AP_current]

	    predict_CO_plus1, predict_AP_plus1 = self.model_predict_future(list_input0)


	    ## predict output_cnn[step+2]
	    DB_plus1 = DB_current
	    SN_plus1 = SN_current
	    output_CO_plus1, output_AP_plus1 = self.math_model(self.input_DB + [DB_current]*2,
	                                                        self.input_SN + [SN_current]*2)
	    
	    list_input1 = [DB_current, DB_plus1, 
	                    SN_current, SN_plus1, 
	                    output_CO_plus1, output_AP_plus1]

	    predict_CO_plus2, predict_AP_plus2 = self.model_predict_future(list_input1)


	    ## predict output_cnn[step+3]
	    DB_plus2 = DB_current
	    SN_plus2 = SN_current
	    output_CO_plus2, output_AP_plus2 = self.math_model(self.input_DB + [DB_current]*3,
	                                                        self.input_SN + [SN_current]*3)
	    
	    list_input2 = [DB_plus1, DB_plus2, 
	                    SN_plus1, SN_plus2, 
	                    output_CO_plus2, output_AP_plus2]

	    predict_CO_plus3, predict_AP_plus3 = self.model_predict_future(list_input2) 


	    ## predict output_cnn[step+4]
	    DB_plus3 = DB_current
	    SN_plus3 = SN_current
	    output_CO_plus3, output_AP_plus3 = self.math_model(self.input_DB + [DB_current]*4,
	                                                        self.input_SN + [SN_current]*4)
	    
	    list_input3 = [DB_plus2, DB_plus3, 
	                    SN_plus2, SN_plus3, 
	                    output_CO_plus3, output_AP_plus3]

	    predict_CO_plus4, predict_AP_plus4 = self.model_predict_future(list_input3) 


	    ## loss function
	    loss = (predict_CO_plus1 - self.r_CO)**2 + (predict_AP_plus1 - self.r_AP)**2 + \
	            (predict_CO_plus2 - self.r_CO)**2 + (predict_AP_plus2 - self.r_AP)**2 + \
	            (predict_CO_plus3 - self.r_CO)**2 + (predict_AP_plus3 - self.r_AP)**2 + \
	            (predict_CO_plus4 - self.r_CO)**2 + (predict_AP_plus4 - self.r_AP)**2 + \
	            0.1*(DB_current-DB_minus1)**2 + 0.1*(SN_current-SN_minus1)**2


	    # print('(DB,SN)=(%.2f, %.2f) | (math_CO, math_AP)=(%.2f, %.2f) | (r_CO, cnn_CO)=(%.2f, %.2f) | (r_AP, cnn_AP)=(%.2f, %.2f) | loss=%.2f' 
	    #       %(DB_current, SN_current, output_CO_current, output_AP_current, self.r_CO, predict_CO_plus1, self.r_AP, predict_AP_plus1, loss))

	    return loss



	#=========================================================#
	def optimize_nelder(self):

	    objective = lambda x: self.calculate_loss(x)

	    result = minimize(objective, 
	                    x0=[0.1, 0.1],
	                    method='nelder-mead',
	                    options={'maxiter':500},
	                    bounds= [(0, 6), (0, 6)])	

	    # evaluate solution
	    solution = result['x']

	    return solution




########################################################
########################################################
def control(MODE_EVALUATE):
	"""
	main script
	"""

	if MODE_EVALUATE == "default":
		NUM_STEPS = 40
	elif MODE_EVALUATE == "interact1":
		NUM_STEPS = 200
	elif MODE_EVALUATE == "combine":
		NUM_STEPS = 300


	#################################
	DB_optimize =  [None]*NUM_STEPS
	SN_optimize =  [None]*NUM_STEPS

	output_desire_CO = [None]*NUM_STEPS
	output_desire_AP = [None]*NUM_STEPS
	output_math_CO =  [None]*NUM_STEPS
	output_cnn_CO =  [None]*NUM_STEPS
	output_math_AP =  [None]*NUM_STEPS
	output_cnn_AP =  [None]*NUM_STEPS

	SENS = [None]*NUM_STEPS

	(SCALER_X, SCALER_Y) = utils.read_from_pickle(PATH_FILE_SCALER)
	model_base = keras.models.load_model(PATH_FILE_MODEL)


	#################################
	if MODE_EVALUATE == "default":

		for step in range(0, NUM_STEPS):
			SENS[step] = 1
			if step <= 20:
				output_desire_CO[step] = 0
				output_desire_AP[step] = 0

			elif 20 < step < 30:
				output_desire_CO[step] = 35*(step-20)/(30-20)
				output_desire_AP[step] = 0

			elif 30 <= step <= 110:
				output_desire_CO[step] = 35
				output_desire_AP[step] = 0

	elif MODE_EVALUATE == "interact1":

		for step in range(0, NUM_STEPS):
			if step <= 20:
				output_desire_CO[step] = 0
				output_desire_AP[step] = 0

			elif 20 < step < 30:
				output_desire_CO[step] = 35*(step-21)/(30-21)
				output_desire_AP[step] = 0

			elif 30 <= step:
				output_desire_CO[step] = 35
				output_desire_AP[step] = 0

		for step in range(0, NUM_STEPS):
			if step <= 20:
				SENS[step] = 1
			elif 21 <= step <= 80:
				SENS[step] = 2
			elif 81 <= step <= 90:
				SENS[step] = (2-1/3)*(91-step)/(91-80) + 1/3
			elif 91 <= step <= 150:
				SENS[step] = 1/3
			elif 151 <= step <= 160:
				SENS[step] = (3-1/3)*(step-150)/(161-150) + 1/3
			elif step >= 161:
				SENS[step] = 3



	elif MODE_EVALUATE == "combine":

		for step in range(0, NUM_STEPS):
			if step <= 20:
				output_desire_CO[step] = 0
				output_desire_AP[step] = 0

			elif 20 < step < 30:
				output_desire_CO[step] = 35*(step-21)/(30-21)
				output_desire_AP[step] = 0

			elif 30 <= step:
				output_desire_CO[step] = 35
				output_desire_AP[step] = 0

		for step in range(0, NUM_STEPS):
			if step <= 20:
				SENS[step] = 1
			elif 21 <= step <= 80:
				SENS[step] = 1

			## 1 up 2
			elif 80 < step < 90:
				SENS[step] = 2 - (2-1)*(90-step)/(90-80)
			elif 90 <= step <= 140:
				SENS[step] = 2

			## 2 down 1/3
			elif 140 < step < 150:
				SENS[step] = 1/3 + (2-1/3)*(150-step)/(150-140)
			elif 150 <= step <= 200:
				SENS[step] = 1/3

			## 1/3 up 3
			elif 200 < step < 210:
				SENS[step] = 3 - (3-1/3)*(210-step)/(210-200)
			elif 210 <= step <= 260:
				SENS[step] = 3

			## 3 down 1
			elif 260 < step < 270:
				SENS[step] = 1 + (3-1)*(270-step)/(270-260)
			elif 270 <= step <= 300:
				SENS[step] = 1

	#################################
	# # PLOT CHECK OUTPUT DESIRE + SENSITIVITIES
	# plt.figure(figsize=(20, 15))

	# plt.subplot(311)
	# plt.plot(output_desire_CO, 'k--', label='desire CO')
	# plt.legend()
	# plt.xlim((0, NUM_STEPS))
	# plt.ylim((-10, 50))

	# plt.subplot(312)
	# plt.plot(output_desire_AP, 'k--', label='desire AP')
	# plt.legend()
	# plt.xlim((0, NUM_STEPS))
	# plt.ylim((-20, 20))

	# plt.subplot(313)
	# plt.plot(SENS, 'k--', label='sensitivities')
	# plt.legend()
	# plt.xlim((0, NUM_STEPS))
	# plt.ylim((0, 5))

	# plt.savefig(PATH_CONTROLLER_CONDITION)
	# plt.close()





	#################################
	for step in range(0, NUM_STEPS-1):

		SENSITIVITIES = (SENS[step], SENS[step], SENS[step], SENS[step])

		if step <= 20:
		## step 1,2,3...19,20 => nothing happend, just randomlize the input DB,SN

			if step == 0:
				DB_optimize[step] = 0
				SN_optimize[step] = 0
			else:
				DB_optimize[step] = 0.1
				SN_optimize[step] = 0.1

			output_math_CO[step] = 0
			output_math_AP[step] = 0

			output_cnn_CO[step] = 0
			output_cnn_AP[step] = 0


		else:
		## starting from step 21

			if step == 21:
				output_cnn_CO[step] = 0
				output_cnn_AP[step] = 0


			## start optimize (DB,SN) from step 21 (t=630s), optimize the loss of step 22,23,24
			s = Controllers(scaler_x=SCALER_X,
							scaler_y=SCALER_Y,
							model=model_base,
							input_DB = DB_optimize[0 : step],
							input_SN = SN_optimize[0 : step],
							r_CO = output_desire_CO[step],
							r_AP = output_desire_AP[step],
							sensitivities = SENSITIVITIES)

			(DB_optimize[step], SN_optimize[step]) = s.optimize_nelder()
	


			## Calculate the Math Response (current step) with optimized DB,SN
			output_math_CO[step], output_math_AP[step] = s.math_model(input_DB = DB_optimize[0:step+1], 
		                                                    		  input_SN = SN_optimize[0:step+1])
	

			# # ## CNN output
			output_cnn_CO[step+1], output_cnn_AP[step+1] = s.model_predict_future(input_list = [DB_optimize[step-1], DB_optimize[step],
							                                                                  SN_optimize[step-1], SN_optimize[step],
							                                                                  output_math_CO[step], output_math_AP[step]])




		# ##########
			print('[step %d], (DB,SN)=(%.2f, %.2f) | (math_CO, math_AP)=(%.2f, %.2f) | (r_CO, cnn_CO)=(%.2f, %.2f) | (r_AP, cnn_AP)=(%.2f, %.2f) | sens= %.2f' 
				%(step, DB_optimize[step], SN_optimize[step], output_math_CO[step], output_math_AP[step], 
					output_desire_CO[step], output_cnn_CO[step+1], output_desire_AP[step], output_cnn_AP[step+1], SENSITIVITIES[0]))

	# print(SN_optimize)
	# utils.save_to_pickle(PATH_CONTROLLER_SAVE, [output_desire_CO, output_math_CO, output_cnn_CO,
	# 										     output_desire_AP, output_math_AP, output_cnn_AP,
	# 										     DB_optimize, SN_optimize, SENS])





def plot_controller(plot_='LifeTech3'):

	from sklearn.metrics import mean_absolute_error

	path = r'D:\00_RITSUMEIKAN UNIVERSITY\1_Fall2021\(4credit) LAB\project_RU_druginfusion\file_save'+'\\'

	###########################################################
	####################### CONTROLLERS #######################

	"""sensitivities changed, with noises +-5 (old version) """
	if plot_ == 'LifeTech1':
		
		[output_desire_CO, output_math_CO, output_cnn_CO,
		output_desire_AP, output_math_AP, output_cnn_AP,
		DB_optimize, SN_optimize, SENS, OFFSET_CO, OFFSET_AP, TIME] = utils.read_from_pickle(path + 'sens_changed_noise_5.pkl')

		plt.figure(figsize=(20, 10))

		plt.subplot(411)
		plt.plot(SENS, 'k--')
		plt.ylabel('Sensitivities', fontsize=15)
		plt.xlim((21, 300))
		plt.ylim((0, 4))
		plt.xticks([])
		plt.yticks(fontsize=12)

		plt.subplot(412)
		plt.plot(output_desire_CO, 'k--', label='ΔCO_desire')
		plt.plot(output_math_CO, 'g-', label='ΔCO_math')
		plt.plot(output_cnn_CO,'g--', label='ΔCO_cnn',  linewidth=1.5)
		plt.ylabel('ΔCO (ml/kg/min)', fontsize=15)
		plt.xlim((21, 300))
		plt.ylim((0, 90))
		plt.xticks([])
		plt.yticks(fontsize=12)
		plt.legend(fontsize=10)

		plt.subplot(413)
		plt.plot(output_desire_AP, 'k--', label='ΔAP_desire')
		plt.plot(output_math_AP, 'b-', label='ΔAP_math')
		plt.plot(output_cnn_AP,'b--', label='ΔAP_cnn',  linewidth=1.5)
		plt.ylabel('ΔAP (mmHg)', fontsize=15)
		plt.xlim((21, 300))
		plt.ylim((-20, 40))
		plt.xticks([])
		plt.yticks([-20, 0, 20, 40], ['','0','20','40'], fontsize=12)
		plt.legend(fontsize=10)


		plt.subplot(414)
		plt.plot(DB_optimize, 'c-', label='DB_optimized')
		plt.plot(SN_optimize, 'm-', label='SN_optimized')
		plt.xlabel('Time (minutes)', fontsize=15)
		plt.ylabel('Infusion (μg/kg/min)', fontsize=15)
		plt.xlim((21, 300))
		plt.ylim((0, 7))
		plt.xticks(range(21,301,20), range(0,140,10), fontsize=12)
		plt.yticks(fontsize=12)
		plt.legend(fontsize=10)

		plt.tight_layout()
		# plt.show()

		plt.savefig(path + 'result_sens_changed_noise_5.png', dpi=700)
		plt.close()

		#############
		# for i in range(0, len(DB_optimize)):
		# 	print('step: %d, (CO, AP) =  (%.2f, %.2f), sens=%.2f' 
		# 		%(i-21, output_math_CO[i], output_math_AP[i], SENS[i]))


		print('step 0-59 (21-80): error (CO, AP) = (%.2f, %.2f)'%(
			mean_absolute_error(output_desire_CO[21: 81], output_math_CO[21:81]),
			mean_absolute_error(output_desire_AP[21: 81], output_math_AP[21:81])))

		print('step 60-119 (81: 140): error (CO, AP) = (%.2f, %.2f)'%(
			mean_absolute_error(output_desire_CO[81: 141], output_math_CO[81:141]),
			mean_absolute_error(output_desire_AP[81: 141], output_math_AP[81:141])))

		print('step 120-179 (141: 200): error (CO, AP) = (%.2f, %.2f)'%(
			mean_absolute_error(output_desire_CO[141: 201], output_math_CO[141:201]),
			mean_absolute_error(output_desire_AP[141: 201], output_math_AP[141:201])))

		print('step 180-239 (201: 260): error (CO, AP) = (%.2f, %.2f)'%(
			mean_absolute_error(output_desire_CO[201: 261], output_math_CO[201:261]),
			mean_absolute_error(output_desire_AP[201: 261], output_math_AP[201:261])))

		print('step 240-277 (261: 298): error (CO, AP) = (%.2f, %.2f)'%(
			mean_absolute_error(output_desire_CO[261: 299], output_math_CO[261:299]),
			mean_absolute_error(output_desire_AP[261: 299], output_math_AP[261:299])))


	###################################################
	"""sensitivities changed, with noises =0 & 5 (official in paper)"""
	if plot_ == 'LifeTech2':
		
		[output_desire_CO, output_math_CO1, output_cnn_CO,
		output_desire_AP, output_math_AP1, output_cnn_AP,
		DB_optimize1, SN_optimize1, SENS] = utils.read_from_pickle(path + 'sens_changed_noise_0.pkl')

		[output_desire_CO, output_math_CO2, output_cnn_CO,
		output_desire_AP, output_math_AP2, output_cnn_AP,
		DB_optimize2, SN_optimize2, SENS, OFFSET_CO, OFFSET_AP, TIME] = utils.read_from_pickle(path + 'sens_changed_noise_5.pkl')


		plt.figure(figsize=(40, 12))

		plt.subplot(411)
		plt.plot(SENS, 'k--', linewidth=4)
		plt.ylabel('Sensitivity', fontsize=24)
		plt.xlim((21, 300))
		plt.ylim((0, 4))
		plt.xticks([])
		plt.yticks([0, 1, 2, 3, 4], [0,1,2,3,4], fontsize=24)
		plt.yticks(fontsize=24)
		plt.text(40, 3, '(a)', fontsize=30)

		plt.subplot(412)
		plt.plot(output_desire_CO, 'k--', label='ΔCO_desire', linewidth=4)
		plt.plot(output_math_CO1, 'g-', label='ΔCO (noise = 0)', linewidth=4)
		plt.plot(output_math_CO2, 'g--', label='ΔCO (noise = +-5)', linewidth=4)
		plt.ylabel('ΔCO (ml/kg/min)', fontsize=24)
		plt.xlim((21, 300))
		plt.ylim((0, 90))
		plt.xticks([])
		plt.yticks([0, 40, 80], [0,40,80], fontsize=24)
		plt.legend(fontsize=18, loc='upper right')
		plt.text(40, 60, '(b)', fontsize=30)

		plt.subplot(413)
		plt.plot(output_desire_AP, 'k--', label='ΔAP_desire', linewidth=4)
		plt.plot(output_math_AP1, 'b-', label='ΔAP (noise = 0)', linewidth=4)
		plt.plot(output_math_AP2, 'b--', label='ΔAP (noise = ±5)', linewidth=4)
		plt.ylabel('ΔAP (mmHg)', fontsize=24)
		plt.xlim((21, 300))
		plt.ylim((-20, 40))
		plt.xticks([])
		plt.yticks([-20, 0, 20, 40], [-20,0,20,40], fontsize=24)
		plt.legend(fontsize=18, loc='upper right')
		plt.text(40, 25, '(c)', fontsize=30)


		plt.subplot(414)
		plt.plot(DB_optimize1, 'c-', label='DB (noise = 0)', linewidth=4)
		plt.plot(DB_optimize2, 'c--', label='DB (noise = ±5)', linewidth=4)
		plt.plot(SN_optimize1, 'm-', label='SN (noise = 0)', linewidth=4)
		plt.plot(SN_optimize2, 'm--', label='SN (noise = ±5)', linewidth=4)
		plt.xlabel('Time (minutes)', fontsize=24)
		plt.ylabel('Infusion (μg/kg/min)', fontsize=24)
		plt.xlim((21, 300))
		plt.ylim((0, 7))
		plt.xticks(range(21,301,20), range(0,140,10), fontsize=24)
		plt.yticks([0,2,4,6], [0,2,4,6], fontsize=24)
		plt.yticks(fontsize=24)
		plt.legend(fontsize=18, loc='upper right')
		plt.text(40, 4, '(d)', fontsize=30)

		plt.tight_layout()
		# plt.show()

		plt.savefig(path + 'result_sens_changed_noise_0&5_ver2.png', dpi=700)
		plt.close()

		#############
		# for i in range(0, len(DB_optimize)):
		# 	print('step: %d, (CO, AP) =  (%.2f, %.2f), sens=%.2f' 
		# 		%(i-21, output_math_CO[i], output_math_AP[i], SENS[i]))

		# print('step 0-59 (21-80): error (CO, AP) = (%.2f, %.2f)'%(
		# 	mean_absolute_error(output_desire_CO[21: 81], output_math_CO[21:81]),
		# 	mean_absolute_error(output_desire_AP[21: 81], output_math_AP[21:81])))

		# print('step 60-119 (81: 140): error (CO, AP) = (%.2f, %.2f)'%(
		# 	mean_absolute_error(output_desire_CO[81: 141], output_math_CO[81:141]),
		# 	mean_absolute_error(output_desire_AP[81: 141], output_math_AP[81:141])))

		# print('step 120-179 (141: 200): error (CO, AP) = (%.2f, %.2f)'%(
		# 	mean_absolute_error(output_desire_CO[141: 201], output_math_CO[141:201]),
		# 	mean_absolute_error(output_desire_AP[141: 201], output_math_AP[141:201])))

		# print('step 180-239 (201: 260): error (CO, AP) = (%.2f, %.2f)'%(
		# 	mean_absolute_error(output_desire_CO[201: 261], output_math_CO[201:261]),
		# 	mean_absolute_error(output_desire_AP[201: 261], output_math_AP[201:261])))

		# print('step 240-277 (261: 298): error (CO, AP) = (%.2f, %.2f)'%(
		# 	mean_absolute_error(output_desire_CO[261: 299], output_math_CO[261:299]),
		# 	mean_absolute_error(output_desire_AP[261: 299], output_math_AP[261:299])))
	

	###################################################
	"""sensitivities changed, with noises =0 & 5 (presentation)"""
	if plot_ == 'LifeTech3':
		
		[output_desire_CO, output_math_CO1, output_cnn_CO,
		output_desire_AP, output_math_AP1, output_cnn_AP,
		DB_optimize1, SN_optimize1, SENS] = utils.read_from_pickle(path + 'sens_changed_noise_0.pkl')

		[output_desire_CO, output_math_CO2, output_cnn_CO,
		output_desire_AP, output_math_AP2, output_cnn_AP,
		DB_optimize2, SN_optimize2, SENS, OFFSET_CO, OFFSET_AP, TIME] = utils.read_from_pickle(path + 'sens_changed_noise_5.pkl')


		plt.figure(figsize=(15, 12))

		plt.subplot(411)
		plt.plot(SENS, 'k--', linewidth=2)
		plt.ylabel('Sensitivity', fontsize=24)
		plt.xlim((21, 300))
		plt.ylim((0, 4))
		plt.xticks([])
		plt.yticks([0, 1, 2, 3, 4], [0,1,2,3,4], fontsize=24)
		plt.yticks(fontsize=24)

		plt.subplot(412)
		plt.plot(output_desire_CO, 'k--', label='ΔCO desire = +35', linewidth=2)
		plt.plot(output_math_CO1, 'g-', label='ΔCO (noise = 0)', linewidth=2)
		plt.plot(output_math_CO2, 'g--', label='ΔCO (noise = +-5)', linewidth=2)
		plt.ylabel('ml/kg/min', fontsize=24)
		plt.xlim((21, 300))
		plt.ylim((0, 90))
		plt.xticks([])
		plt.yticks([0, 40, 80], [0,40,80], fontsize=24)
		plt.legend(fontsize=15, loc='upper right')

		plt.subplot(413)
		plt.plot(output_desire_AP, 'k--', label='ΔAP desire = 0', linewidth=2)
		plt.plot(output_math_AP1, 'b-', label='ΔAP (noise = 0)', linewidth=2)
		plt.plot(output_math_AP2, 'b--', label='ΔAP (noise = ±5)', linewidth=2)
		plt.ylabel('mmHg', fontsize=24)
		plt.xlim((21, 300))
		plt.ylim((-20, 40))
		plt.xticks([])
		plt.yticks([-20, 0, 20, 40], [-20,0,20,40], fontsize=24)
		plt.legend(fontsize=15, loc='upper right')

		plt.subplot(414)
		plt.plot(DB_optimize1, 'c-', label='DB (noise = 0)', linewidth=2)
		plt.plot(DB_optimize2, 'c--', label='DB (noise = ±5)', linewidth=2)
		plt.plot(SN_optimize1, 'm-', label='SN (noise = 0)', linewidth=2)
		plt.plot(SN_optimize2, 'm--', label='SN (noise = ±5)', linewidth=2)
		plt.xlabel('Time (minutes)', fontsize=24)
		plt.ylabel('μg/kg/min', fontsize=24)
		plt.xlim((21, 300))
		plt.ylim((0, 7))
		plt.xticks(range(21,301,20), range(0,140,10), fontsize=24)
		plt.yticks([0,2,4,6], [0,2,4,6], fontsize=24)
		plt.yticks(fontsize=24)
		plt.legend(fontsize=15, loc='upper right')
		# plt.text(40, 4, 'Drug Infusion Rate', fontsize=20)

		plt.tight_layout()
		# plt.show()

		plt.savefig(path + 'result_sens_changed_noise_0&5_presentation.png', dpi=700)
		plt.close()




	###################################################	
	"""sensitivities changed, with noises (0, +-10, +-20), plot noise, calculate time"""
	if plot_ == 'journal1':
		
		[output_desire_CO, output_math_CO1, output_cnn_CO,
		output_desire_AP, output_math_AP1, output_cnn_AP,
		DB_optimize1, SN_optimize1, SENS] = utils.read_from_pickle(path + 'sens_changed_noise_0.pkl')

		[output_desire_CO, output_math_CO2, output_cnn_CO,
		output_desire_AP, output_math_AP2, output_cnn_AP,
		DB_optimize2, SN_optimize2, SENS, OFFSET_CO2, OFFSET_AP2, TIME2] = utils.read_from_pickle(path + 'sens_changed_noise_10.pkl')

		[output_desire_CO, output_math_CO3, output_cnn_CO,
		output_desire_AP, output_math_AP3, output_cnn_AP,
		DB_optimize3, SN_optimize3, SENS, OFFSET_CO3, OFFSET_AP3, TIME3] = utils.read_from_pickle(path + 'sens_changed_noise_20.pkl')


		plt.figure(figsize=(18, 10))

		plt.subplot(431)
		plt.plot(SENS, 'k--')
		plt.ylabel('Sensitivities', fontsize=15)
		plt.xlim((21, 300))
		plt.ylim((0, 4))
		plt.xticks([])
		plt.yticks(fontsize=12)

		plt.subplot(432)
		plt.plot(SENS, 'k--')
		plt.xlim((21, 300))
		plt.ylim((0, 4))
		plt.xticks([])
		plt.yticks(fontsize=12)

		plt.subplot(433)
		plt.plot(SENS, 'k--')
		plt.xlim((21, 300))
		plt.ylim((0, 4))
		plt.xticks([])
		plt.yticks(fontsize=12)

		## CO
		plt.subplot(434)
		plt.plot(output_desire_CO, 'k--', label='ΔCO_desire')
		plt.plot(output_math_CO1, 'g-', label='noise +- 0')
		plt.ylabel('ΔCO (ml/kg/min)', fontsize=15)
		plt.xlim((21, 300))
		plt.ylim((0, 100))
		plt.xticks([])
		plt.yticks(fontsize=12)
		plt.legend(loc='upper right', fontsize=8)

		plt.subplot(435)
		plt.plot(output_desire_CO, 'k--')
		plt.plot(output_math_CO2, 'g-', label='noise +- 10')
		plt.xlim((21, 300))
		plt.ylim((0, 100))
		plt.xticks([])
		plt.yticks(fontsize=12)
		plt.legend(loc='upper right', fontsize=8)

		plt.subplot(436)
		plt.plot(output_desire_CO, 'k--')
		plt.plot(output_math_CO3, 'g-', label='noise +- 20')
		plt.xlim((21, 300))
		plt.ylim((0, 100))
		plt.xticks([])
		plt.yticks(fontsize=12)
		plt.legend(loc='upper right', fontsize=8)

		## AP
		plt.subplot(437)
		plt.plot(output_desire_AP, 'k--', label='ΔAP_desire')
		plt.plot(output_math_AP1, 'b-', label='noise +- 0')
		plt.ylabel('ΔAP (mmHg)', fontsize=15)
		plt.xlim((21, 300))
		plt.ylim((-40, 40))
		plt.xticks([])
		plt.yticks(fontsize=12)
		plt.legend(loc='upper right', fontsize=8)

		plt.subplot(438)
		plt.plot(output_desire_AP, 'k--')
		plt.plot(output_math_AP2, 'b-', label='noise +- 10')
		plt.xlim((21, 300))
		plt.ylim((-40, 40))
		plt.xticks([])
		plt.yticks(fontsize=12)
		plt.legend(loc='upper right', fontsize=8)

		plt.subplot(439)
		plt.plot(output_desire_AP, 'k--')
		plt.plot(output_math_AP3, 'b-', label='noise +- 20')
		plt.xlim((21, 300))
		plt.ylim((-40, 40))
		plt.xticks([])
		plt.yticks(fontsize=12)
		plt.legend(loc='upper right', fontsize=8)
		# plt.yticks([-20, 0, 20, 40], ['','0','20','40'], fontsize=12)

		##
		plt.subplot(4,3,10)
		plt.plot(DB_optimize1, 'c-', label='DB')
		plt.plot(SN_optimize1, 'm-', label='SN')
		plt.xlabel('Time (minutes)', fontsize=15)
		plt.ylabel('Infusion (μg/kg/min)', fontsize=15)
		plt.xlim((21, 300))
		plt.ylim((0, 7))
		plt.xticks(range(21,301,60), range(0,140,30), fontsize=12)
		plt.yticks(fontsize=12)
		plt.legend(fontsize=10)

		plt.subplot(4,3,11)
		plt.plot(DB_optimize2, 'c-', label='DB_optimized')
		plt.plot(SN_optimize2, 'm-', label='SN_optimized')
		plt.xlabel('Time (minutes)', fontsize=15)
		plt.xlim((21, 300))
		plt.ylim((0, 7))
		plt.xticks(range(21,301,60), range(0,140,30), fontsize=12)
		plt.yticks(fontsize=12)
		# plt.legend(fontsize=10)

		plt.subplot(4,3,12)
		plt.plot(DB_optimize3, 'c-', label='DB_optimized')
		plt.plot(SN_optimize3, 'm-', label='SN_optimized')
		plt.xlabel('Time (minutes)', fontsize=15)
		plt.xlim((21, 300))
		plt.ylim((0, 7))
		plt.xticks(range(21,301,60), range(0,140,30), fontsize=12)
		plt.yticks(fontsize=12)
		# plt.legend(fontsize=10)

		plt.tight_layout()
		# plt.show()
		
		plt.savefig(path + 'result_sens_changed_noise_0&10&20.png', dpi=700)
		plt.close()

	###################################################

	"""ACUTE DISTURBANCE WITH HIGH SENSITIVITIES"""
	if plot_ == 'journal2':

		## 
		[output_desire_CO, output_math_CO1, output_cnn_CO,
		output_desire_AP, output_math_AP1, output_cnn_AP,
		DB_optimize1, SN_optimize1, SENS] = utils.read_from_pickle(path + 'acutedisturbance_limited_1.pkl')

		[output_desire_CO, output_math_CO2, output_cnn_CO,
		output_desire_AP, output_math_AP2, output_cnn_AP,
		DB_optimize2, SN_optimize2, SENS] = utils.read_from_pickle(path + 'acutedisturbance_limited_2.pkl')

		[output_desire_CO, output_math_CO3, output_cnn_CO,
		output_desire_AP, output_math_AP3, output_cnn_AP,
		DB_optimize3, SN_optimize3, SENS] = utils.read_from_pickle(path + 'acutedisturbance_limited_3.pkl')

		NUM_STEPS = 130
		OFFSET_CO1  = [None]*NUM_STEPS
		OFFSET_CO2  = [None]*NUM_STEPS
		OFFSET_CO3  = [None]*NUM_STEPS
		OFFSET_AP1 = [None]*NUM_STEPS
		OFFSET_AP2 = [None]*NUM_STEPS
		OFFSET_AP3 = [None]*NUM_STEPS

		for step in range(0, NUM_STEPS):
			SENS[step] = 3
			if step <= 60:
				OFFSET_CO1[step] = 0
				OFFSET_CO2[step] = 0
				OFFSET_CO3[step] = 0
				OFFSET_AP1[step] = 0
				OFFSET_AP2[step] = 0
				OFFSET_AP3[step] = 0

			## level 1 [10mins]
			if 60 < step < 80:
				OFFSET_CO1[step] = (-50) + ((0)-(-50)) * (80-step)/(80-60) # 0 -> -50
				OFFSET_AP1[step] = (50) - ((50)-(0)) * (80-step)/(80-60) # 0 -> +50 
			elif 80 <= step <= 130:
				OFFSET_CO1[step] = (-50)
				OFFSET_AP1[step] = 50

			## level 2 [5mins]
			if 60 < step < 70:
				OFFSET_CO2[step] = (-50) + ((0)-(-50)) * (70-step)/(70-60) # 0 -> -50
				OFFSET_AP2[step] = (50) - ((50)-(0)) * (70-step)/(70-60) # 0 -> +50 
			elif 70 <= step <= 130:
				OFFSET_CO2[step] = (-50)
				OFFSET_AP2[step] = 50

			# level 3 [0mins]
			if 60 < step <= 130:
				OFFSET_CO3[step] = (-50)
				OFFSET_AP3[step] = 50

		##

		# plt.figure(figsize=(40, 12))

		# ## 
		# plt.subplot(321)
		# plt.plot(OFFSET_CO1, 'g-', label='level 1 (10 min)', linewidth=6)
		# plt.plot(OFFSET_CO2, 'g--', label='level 2 (5 min)', linewidth=6)
		# plt.plot(OFFSET_CO3, 'g:', label='level 3 (0 min)', linewidth=6)
		# plt.ylabel('ΔCO Disturbance', fontsize=24)
		# plt.xlim((21, NUM_STEPS))
		# plt.xticks([])
		# plt.yticks([0,-50], [0,-50], fontsize=24)
		# plt.legend(fontsize=18)

		# plt.subplot(322)
		# plt.plot(OFFSET_AP1, 'b-', label='level 1 (10 min)', linewidth=6)
		# plt.plot(OFFSET_AP2, 'b--', label='level 2 (5 min)', linewidth=6)
		# plt.plot(OFFSET_AP3, 'b:', label='level 3 (0 min)', linewidth=6)
		# plt.ylabel('ΔAP Disturbance ', fontsize=24)
		# plt.xlim((21, NUM_STEPS))
		# plt.xticks([])
		# plt.yticks([0,50], [0,50], fontsize=24)
		# plt.legend(fontsize=18)

		# ##
		# plt.subplot(323)
		# plt.plot(output_desire_CO, 'k--', label='desire ΔCO', linewidth=6)
		# plt.plot(output_math_CO1, 'g-', label='ΔCO (level 1)', linewidth=6)
		# plt.plot(output_math_CO2, 'g--', label='ΔCO (level 2)', linewidth=6)
		# plt.plot(output_math_CO3, 'g:', label='ΔCO (level 3)', linewidth=6)		
		# plt.ylabel('Observed ΔCO (ml/kg/min)', fontsize=24)
		# plt.xlim((21, NUM_STEPS))
		# plt.ylim((-60, 60))
		# plt.xticks([])
		# plt.yticks([-60, -30, 0, 30, 60], [-60, -30, 0, 30, 60], fontsize=24)
		# plt.legend(loc='lower right', fontsize=18)

		# plt.subplot(324)
		# plt.plot(output_desire_AP, 'k--', label='desire ΔAP', linewidth=6)
		# plt.plot(output_math_AP1, 'b-', label='ΔAP (level 1)', linewidth=6)
		# plt.plot(output_math_AP2, 'b--', label='ΔAP (level 2)', linewidth=6)
		# plt.plot(output_math_AP3, 'b:', label='ΔAP (level 3)', linewidth=6)
		# plt.ylabel('Observed ΔAP (mmHg)', fontsize=24)
		# plt.xlim((21, NUM_STEPS))
		# plt.ylim((-60, 60))
		# plt.xticks([])
		# plt.yticks([-60, -30, 0, 30, 60], [-60, -30, 0, 30, 60], fontsize=24)
		# plt.legend(loc='lower right', fontsize=18)

		# ##
		# plt.subplot(325)
		# plt.plot(DB_optimize1, 'c-', label='DB (level 1)', linewidth=6)
		# plt.plot(DB_optimize2, 'c--', label='DB (level 2)', linewidth=6)
		# plt.plot(DB_optimize3, 'c:', label='DB (level 3)', linewidth=6)
		# plt.xlabel('Time (minutes)', fontsize=24)
		# plt.ylabel('DB (μg/kg/min)', fontsize=24)
		# plt.xlim((21, NUM_STEPS))
		# plt.ylim((0, 2))
		# plt.xticks(range(21,131,20), range(0,60,10), fontsize=24)
		# plt.yticks([0,1,2], [0,1,2], fontsize=24)
		# plt.legend(loc='upper left', fontsize=18)

		# plt.subplot(326)
		# plt.plot(SN_optimize1, 'm-', label='SN (level 1)', linewidth=6)
		# plt.plot(SN_optimize2, 'm--', label='SN (level 2)', linewidth=6)
		# plt.plot(SN_optimize3, 'm:', label='SN (level 3)', linewidth=6)
		# plt.xlabel('Time (minutes)', fontsize=24)
		# plt.ylabel('SN (μg/kg/min)', fontsize=24)
		# plt.xlim((21, NUM_STEPS))
		# plt.ylim((0, 7))
		# plt.xticks(range(21,131,20), range(0,60,10), fontsize=24)
		# plt.yticks([0,2,4,6], [0,2,4,6], fontsize=24)
		# plt.legend(loc='upper left', fontsize=18)

		# plt.tight_layout()
		# # plt.show()

		# plt.savefig(path + 'result_acutedisturbanceHighSens_limited.png', dpi=700)
		# plt.close()


		################

		# for i in range(0, NUM_STEPS):
		# 	print('step: %d, (CO, AP) =  (%.2f, %.2f), sens=%.2f' 
		# 		%(i-21, output_math_CO3[i], output_math_AP3[i], SENS[i]))

		print('(level 1) step 64-100 (85-121): error (CO, AP) = (%.2f, %.2f)'%(
			mean_absolute_error(output_desire_CO[85: 121], output_math_CO1[85:121]),
			mean_absolute_error(output_desire_AP[85: 121], output_math_AP1[85:121])))
		print('(level 2) step 64-100 (85-121): error (CO, AP) = (%.2f, %.2f)'%(
			mean_absolute_error(output_desire_CO[85: 121], output_math_CO2[85:121]),
			mean_absolute_error(output_desire_AP[85: 121], output_math_AP2[85:121])))
		print('(level 3) step 64-100 (85-121): error (CO, AP) = (%.2f, %.2f)'%(
			mean_absolute_error(output_desire_CO[85: 121], output_math_CO3[85:121]),
			mean_absolute_error(output_desire_AP[85: 121], output_math_AP3[85:121])))


		# print('(level 1) step 40-60 (61-81): error (CO, AP) = (%.2f, %.2f)'%(
		# 	mean_absolute_error(output_desire_CO[61: 81], output_math_CO1[61:81]),
		# 	mean_absolute_error(output_desire_AP[61: 81], output_math_AP1[61:81])))

		# print('(level 2) step 40-50 (61-71): error (CO, AP) = (%.2f, %.2f)'%(
		# 	mean_absolute_error(output_desire_CO[61: 71], output_math_CO2[61:71]),
		# 	mean_absolute_error(output_desire_AP[61: 71], output_math_AP2[61:71])))

		# print('(level 3) step 0-59 (21-80): error (CO, AP) = (%.2f, %.2f)'%(
		# 	mean_absolute_error(output_desire_CO[21: 81], output_math_CO[21:81]),
		# 	mean_absolute_error(output_desire_AP[21: 81], output_math_AP[21:81])))

	###############################################
	"""## HUGE DISTURBANCE"""
	if plot_ == 'journal3':

		[output_desire_CO, output_math_CO1, output_cnn_CO,
		output_desire_AP, output_math_AP1, output_cnn_AP,
		DB_optimize1, SN_optimize1, SENS, OFFSET_CO, OFFSET_AP, TIME1] = utils.read_from_pickle(path + 'hugedisturbance_unlimited1.pkl')

		[output_desire_CO, output_math_CO2, output_cnn_CO,
		output_desire_AP, output_math_AP2, output_cnn_AP,
		DB_optimize2, SN_optimize2, SENS, OFFSET_CO, OFFSET_AP, TIME2] = utils.read_from_pickle(path + 'hugedisturbance_unlimited2.pkl')

		[output_desire_CO, output_math_CO3, output_cnn_CO,
		output_desire_AP, output_math_AP3, output_cnn_AP,
		DB_optimize3, SN_optimize3, SENS, OFFSET_CO, OFFSET_AP, TIME3] = utils.read_from_pickle(path + 'hugedisturbance_unlimited3.pkl')
		
		NUM_STEPS = 130
		OFFSET_CO1  = [None]*NUM_STEPS
		OFFSET_CO2  = [None]*NUM_STEPS
		OFFSET_CO3  = [None]*NUM_STEPS
		OFFSET_AP1 = [None]*NUM_STEPS
		OFFSET_AP2 = [None]*NUM_STEPS
		OFFSET_AP3 = [None]*NUM_STEPS

		for step in range(0, NUM_STEPS):
			SENS[step] = 1
			if step <= 60:
				OFFSET_CO1[step] = 0
				OFFSET_CO2[step] = 0
				OFFSET_CO3[step] = 0
				OFFSET_AP1[step] = 0
				OFFSET_AP2[step] = 0
				OFFSET_AP3[step] = 0

			## level 1
			if 60 < step < 70:
				OFFSET_CO1[step] = (-20) + ((0)-(-20)) * (70-step)/(70-60) # 0 -> -20
				OFFSET_AP1[step] = (20) - ((20)-(0)) * (70-step)/(70-60) # 0 -> +20
			elif 70 <= step <= 130:
				OFFSET_CO1[step] = (-20)
				OFFSET_AP1[step] = 20

			## level 2
			if 60 < step < 70:
				OFFSET_CO2[step] = (-50) + ((0)-(-50)) * (70-step)/(70-60) # 0 -> -50
				OFFSET_AP2[step] = (50) - ((50)-(0)) * (70-step)/(70-60) # 0 -> +50
			elif 70 <= step <= 130:
				OFFSET_CO2[step] = (-50)
				OFFSET_AP2[step] = 50

			## level 3
			if 60 < step < 70:
				OFFSET_CO3[step] = (-100) + ((0)-(-100)) * (70-step)/(70-60) # 0 -> -100
				OFFSET_AP3[step] = (100) - ((100)-(0)) * (70-step)/(70-60) # 0 -> +100
			elif 70 <= step <= 130:
				OFFSET_CO3[step] = (-100)
				OFFSET_AP3[step] = 100

		##

		plt.figure(figsize=(20, 12))

		## 
		plt.subplot(321)
		plt.plot(OFFSET_CO1, 'g-', label='level 1 (-25)')
		plt.plot(OFFSET_CO2, 'g--', label='level 2 (-50)')
		plt.plot(OFFSET_CO3, 'g:', label='level 3 (-100)')
		plt.ylabel('Disturbance in ΔCO', fontsize=15)
		plt.xlim((21, NUM_STEPS))
		plt.xticks([])
		plt.yticks([0,-50, -100], [0,-50, -100], fontsize=12)
		plt.legend(fontsize=10)

		plt.subplot(322)
		plt.plot(OFFSET_AP1, 'b-', label='level 1 (+25)')
		plt.plot(OFFSET_AP2, 'b--', label='level 2 (+50)')
		plt.plot(OFFSET_AP3, 'b:', label='level 3 (+100)')
		plt.ylabel('Disturbance in ΔAP ', fontsize=15)
		plt.xlim((21, NUM_STEPS))
		plt.xticks([])
		plt.yticks([0,50,100], [0,50,100], fontsize=12)
		plt.legend(fontsize=10)

		##
		plt.subplot(323)
		plt.plot(output_desire_CO, 'k--', label='desire ΔCO')
		plt.plot(output_math_CO1, 'g-', label='ΔCO (level 1)')
		plt.plot(output_math_CO2, 'g--', label='ΔCO (level 2)')
		plt.plot(output_math_CO3, 'g:', label='ΔCO (level 3)')		
		plt.ylabel('Observed ΔCO (ml/kg/min)', fontsize=15)
		plt.xlim((21, NUM_STEPS))
		plt.ylim((-100, 60))
		plt.xticks([])
		plt.yticks([-100, -50, 0, 50], [-100,-50,0,50], fontsize=12)
		plt.legend(loc='lower right', fontsize=10)

		plt.subplot(324)
		plt.plot(output_desire_AP, 'k--', label='desire ΔAP')
		plt.plot(output_math_AP1, 'b-', label='ΔAP (level 1)')
		plt.plot(output_math_AP2, 'b--', label='ΔAP (level 2)')
		plt.plot(output_math_AP3, 'b:', label='ΔAP (level 3)')	
		plt.ylabel('Observed ΔAP (mmHg)', fontsize=15)
		plt.xlim((21, NUM_STEPS))
		plt.ylim((-60, 100))
		plt.xticks([])
		plt.yticks([-50, 0, 50, 100], [-50,0,50,100], fontsize=12)
		plt.legend(loc='lower right', fontsize=10)

		##
		plt.subplot(325)
		plt.plot(DB_optimize1, 'c-', label='DB (level 1)')
		plt.plot(DB_optimize2, 'c--', label='DB (level 2)')
		plt.plot(DB_optimize3, 'c:', label='DB (level 3)')
		plt.xlabel('Time (minutes)', fontsize=15)
		plt.ylabel('DB (μg/kg/min)', fontsize=15)
		plt.xlim((21, NUM_STEPS))
		plt.ylim((0, 3))
		plt.xticks(range(21,131,20), range(0,60,10), fontsize=12)
		# plt.yticks([0,2,4,6], [0,2,4,6], fontsize=12)
		plt.yticks(fontsize=12)
		plt.legend(loc='upper left', fontsize=10)

		plt.subplot(326)
		plt.plot(SN_optimize1, 'm-', label='SN (level 1)')
		plt.plot(SN_optimize2, 'm--', label='SN (level 2)')
		plt.plot(SN_optimize3, 'm:', label='SN (level 3)')
		plt.xlabel('Time (minutes)', fontsize=15)
		plt.ylabel('SN (μg/kg/min)', fontsize=15)
		plt.xlim((21, NUM_STEPS))
		plt.ylim((0, 30))
		plt.xticks(range(21,131,20), range(0,60,10), fontsize=12)
		# plt.yticks([0,2,4,6], [0,2,4,6], fontsize=12)
		plt.yticks(fontsize=12)
		plt.legend(loc='upper left', fontsize=10)

		plt.tight_layout()
		# plt.show()

		plt.savefig(path + 'result_hugeDisturbance_unlimited.png', dpi=700)
		plt.close()



####################################
####################################







if __name__ == "__main__":
	cnn.flow_training(re_define_inputs=False, 
						plot_inputs=False, 
						re_train=False, 
						plot_loss=False, 
						plot_test=True)

	# control(MODE_EVALUATE="default")

	# plot_controller()