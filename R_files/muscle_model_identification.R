# Time series analysis with R
library(R.matlab)
library(signal)
library(pracma)
library(abind)
library(stats)

############################################
Fs <- 100
stim_Fs <- 20

hand_data <- readMat("Data/PhD_Dissertation//NJBT_pilot_data/NJBT_ses1_stim_block4.mat",fixNames = "F")
muscle_io_data <- hand_data$NJBT_ses1_stim_block3[c((50*Fs):(70*Fs)),c(1,3,4)]
#force.ts <- ts(data = muscle_io_data[,2],start = 0.00,frequency = 100)
#plot(stl(x = force.ts,s.window = "periodic"))
plot(muscle_io_data[,1],muscle_io_data[,3],type = "l",col="red",ylim = c(0,1300))
lines(muscle_io_data[,1],muscle_io_data[,2])
a1 <- 1
a2 <- 0.0436
b1 <- 0.6
b2 <- 0.9
v_k <- filter(filt = c(0,a2),a = c(1,-1*a1),x = muscle_io_data[,3])
lines(muscle_io_data[,1],v_k,col="blue",ylim = c(0,1300))
w_k <- v_k/(1 + v_k)
lines(muscle_io_data[,1],w_k,col="green",ylim = c(0,1300))
