
############################
# error rate
############################

# 3 con 5 hid
mc_evalu_3con_5hid_1t_error <- 1-mc_evalu_3con_5hid_1t[,2]
mc_evalu_3con_5hid_2n_error <- 1-mc_evalu_3con_5hid_2n[,2]
mc_evalu_3con_5hid_3r_error <- 1-mc_evalu_3con_5hid_3r[,2]
mc_evalu_3con_5hid_4t_error <- 1-mc_evalu_3con_5hid_4t[,2]
mc_evalu_3con_5hid_5t_error <- 1-mc_evalu_3con_5hid_5t[,2]
mc_evalu_3con_5hid_6t_error <- 1-mc_evalu_3con_5hid_6t[,2]
mc_evalu_3con_5hid_7t_error <- 1-mc_evalu_3con_5hid_7t[,2]
mc_evalu_3con_5hid_8t_error <- 1-mc_evalu_3con_5hid_8t[,2]
mc_evalu_3con_5hid_9t_error <- 1-mc_evalu_3con_5hid_9t[,2]
mc_evalu_3con_5hid_10t_error <- 1-mc_evalu_3con_5hid_10t[,2]

all_error_3con_5hid_data <- as.data.frame(cbind(MC_Samples,mc_evalu_3con_5hid_1t_error,
                                                mc_evalu_3con_5hid_2n_error,
                                                mc_evalu_3con_5hid_3r_error,
                                                mc_evalu_3con_5hid_4t_error,
                                                mc_evalu_3con_5hid_5t_error,
                                                mc_evalu_3con_5hid_6t_error,
                                                mc_evalu_3con_5hid_7t_error,
                                                mc_evalu_3con_5hid_8t_error,
                                                mc_evalu_3con_5hid_9t_error,
                                                mc_evalu_3con_5hid_10t_error))


mean_error_3con_5hid <- apply(all_error_3con_5hid_data[,-1],1,mean) #50 output
sd_error_3con_5hid <- apply(all_error_3con_5hid_data[,-1],1,sd) #50 output
upper_3con_5hid <- mean_error_3con_5hid + sd_error_3con_5hid
lower_3con_5hid <- mean_error_3con_5hid - sd_error_3con_5hid



ggplot(data=all_error_3con_5hid_data)+
  geom_errorbar(aes(x = MC_Samples,ymin = lower_3con_5hid ,ymax = upper_3con_5hid),color="#00688b")+
  geom_point(aes(x = MC_Samples,y = mean_error_3con_5hid),size = 1,shape = 21, fill = "blue")+
  scale_y_continuous(labels = scales::percent)+
  geom_line(aes(x = MC_Samples,y = mean_error_3con_5hid),color = "#0f0f43")+
  geom_smooth(aes(x = MC_Samples,y = mean_error_3con_5hid),size = 0.1,color="#8906d2",lty="dotted")+
  labs(x=" Number of MC Samples",
       y="Error Percentage")+
  theme_bw()





# 3 con 4 hid
mc_evalu_3con_4hid_1t_error <- 1-mc_evalu_3con_4hid_1t[,2]
mc_evalu_3con_4hid_2n_error <- 1-mc_evalu_3con_4hid_2n[,2]
mc_evalu_3con_4hid_3r_error <- 1-mc_evalu_3con_4hid_3r[,2]
mc_evalu_3con_4hid_4t_error <- 1-mc_evalu_3con_4hid_4t[,2]
mc_evalu_3con_4hid_5t_error <- 1-mc_evalu_3con_4hid_5t[,2]
mc_evalu_3con_4hid_6t_error <- 1-mc_evalu_3con_4hid_6t[,2]
mc_evalu_3con_4hid_7t_error <- 1-mc_evalu_3con_4hid_7t[,2]
mc_evalu_3con_4hid_8t_error <- 1-mc_evalu_3con_4hid_8t[,2]
mc_evalu_3con_4hid_9t_error <- 1-mc_evalu_3con_4hid_9t[,2]
mc_evalu_3con_4hid_10t_error <- 1-mc_evalu_3con_4hid_10t[,2]



all_error_3con_4hid_data <- as.data.frame(cbind(MC_Samples,mc_evalu_3con_4hid_1t_error,
                                                mc_evalu_3con_4hid_2n_error,
                                                mc_evalu_3con_4hid_3r_error,
                                                mc_evalu_3con_4hid_4t_error,
                                                mc_evalu_3con_4hid_5t_error,
                                                mc_evalu_3con_4hid_6t_error,
                                                mc_evalu_3con_4hid_7t_error,
                                                mc_evalu_3con_4hid_8t_error,
                                                mc_evalu_3con_4hid_9t_error,
                                                mc_evalu_3con_4hid_10t_error))


mean_error_3con_4hid <- apply(all_error_3con_4hid_data[,-1],1,mean) #50 output
sd_error_3con_4hid <- apply(all_error_3con_4hid_data[,-1],1,sd) #50 output
upper_3con_4hid <- mean_error_3con_4hid + sd_error_3con_4hid
lower_3con_4hid <- mean_error_3con_4hid - sd_error_3con_4hid



ggplot(data=all_error_3con_4hid_data)+
  geom_errorbar(aes(x = MC_Samples,ymin = lower_3con_4hid ,ymax = upper_3con_4hid),color="#00688b")+
  geom_point(aes(x = MC_Samples,y = mean_error_3con_4hid),size = 1,shape = 21, fill = "blue")+
  scale_y_continuous(labels = scales::percent)+
  geom_line(aes(x = MC_Samples,y = mean_error_3con_4hid),color = "#0f0f43")+
  geom_smooth(aes(x = MC_Samples,y = mean_error_3con_4hid),size = 0.1,color="#8906d2",lty="dotted")+
  labs(x=" Number of MC Samples",
       y="Error Percentage")+
  theme_bw()




# 4 con 5 hid
mc_evalu_4con_5hid_1t_error <- 1-mc_evalu_4con_5hid_1t[,2]
mc_evalu_4con_5hid_2n_error <- 1-mc_evalu_4con_5hid_2n[,2]
mc_evalu_4con_5hid_3r_error <- 1-mc_evalu_4con_5hid_3r[,2]
mc_evalu_4con_5hid_4t_error <- 1-mc_evalu_4con_5hid_4t[,2]
mc_evalu_4con_5hid_5t_error <- 1-mc_evalu_4con_5hid_5t[,2]
mc_evalu_4con_5hid_6t_error <- 1-mc_evalu_4con_5hid_6t[,2]
mc_evalu_4con_5hid_7t_error <- 1-mc_evalu_4con_5hid_7t[,2]
mc_evalu_4con_5hid_8t_error <- 1-mc_evalu_4con_5hid_8t[,2]
mc_evalu_4con_5hid_9t_error <- 1-mc_evalu_4con_5hid_9t[,2]
mc_evalu_4con_5hid_10t_error <- 1-mc_evalu_4con_5hid_10t[,2]

all_error_4con_5hid_data <- as.data.frame(cbind(MC_Samples,mc_evalu_4con_5hid_1t_error,
                                                mc_evalu_4con_5hid_2n_error,
                                                mc_evalu_4con_5hid_3r_error,
                                                mc_evalu_4con_5hid_4t_error,
                                                mc_evalu_4con_5hid_5t_error,
                                                mc_evalu_4con_5hid_6t_error,
                                                mc_evalu_4con_5hid_7t_error,
                                                mc_evalu_4con_5hid_8t_error,
                                                mc_evalu_4con_5hid_9t_error,
                                                mc_evalu_4con_5hid_10t_error))


mean_error_4con_5hid <- apply(all_error_4con_5hid_data[,-1],1,mean) #50 output
sd_error_4con_5hid <- apply(all_error_4con_5hid_data[,-1],1,sd) #50 output
upper_4con_5hid <- mean_error_4con_5hid + sd_error_4con_5hid
lower_4con_5hid <- mean_error_4con_5hid - sd_error_4con_5hid



ggplot(data=all_error_4con_5hid_data)+
  geom_errorbar(aes(x = MC_Samples,ymin = lower_4con_5hid ,ymax = upper_4con_5hid),color="#00688b")+
  geom_point(aes(x = MC_Samples,y = mean_error_4con_5hid),size = 1,shape = 21, fill = "blue")+
  scale_y_continuous(labels = scales::percent)+
  geom_line(aes(x = MC_Samples,y = mean_error_4con_5hid),color = "#0f0f43")+
  geom_smooth(aes(x = MC_Samples,y = mean_error_4con_5hid),size = 0.1,color="#8906d2",lty="dotted")+
  labs(x=" Number of MC Samples",
       y="Error Percentage")+
  theme_bw()




# 4 con 4 hid
mc_evalu_4con_4hid_1t_error <- 1-mc_evalu_4con_4hid_1t[,2]
mc_evalu_4con_4hid_2n_error <- 1-mc_evalu_4con_4hid_2n[,2]
mc_evalu_4con_4hid_3r_error <- 1-mc_evalu_4con_4hid_3r[,2]
mc_evalu_4con_4hid_4t_error <- 1-mc_evalu_4con_4hid_4t[,2]
mc_evalu_4con_4hid_5t_error <- 1-mc_evalu_4con_4hid_5t[,2]
mc_evalu_4con_4hid_6t_error <- 1-mc_evalu_4con_4hid_6t[,2]
mc_evalu_4con_4hid_7t_error <- 1-mc_evalu_4con_4hid_7t[,2]
mc_evalu_4con_4hid_8t_error <- 1-mc_evalu_4con_4hid_8t[,2]
mc_evalu_4con_4hid_9t_error <- 1-mc_evalu_4con_4hid_9t[,2]
mc_evalu_4con_4hid_10t_error <- 1-mc_evalu_4con_4hid_10t[,2]

all_error_4con_4hid_data <- as.data.frame(cbind(MC_Samples,mc_evalu_4con_4hid_1t_error,
                                                mc_evalu_4con_4hid_2n_error,
                                                mc_evalu_4con_4hid_3r_error,
                                                mc_evalu_4con_4hid_4t_error,
                                                mc_evalu_4con_4hid_5t_error,
                                                mc_evalu_4con_4hid_6t_error,
                                                mc_evalu_4con_4hid_7t_error,
                                                mc_evalu_4con_4hid_8t_error,
                                                mc_evalu_4con_4hid_9t_error,
                                                mc_evalu_4con_4hid_10t_error))


mean_error_4con_4hid <- apply(all_error_4con_4hid_data[,-1],1,mean) #50 output
sd_error_4con_4hid <- apply(all_error_4con_4hid_data[,-1],1,sd) #50 output
upper_4con_4hid <- mean_error_4con_4hid + sd_error_4con_4hid
lower_4con_4hid <- mean_error_4con_4hid - sd_error_4con_4hid



ggplot(data=all_error_4con_4hid_data)+
  geom_errorbar(aes(x = MC_Samples,ymin = lower_4con_4hid ,ymax = upper_4con_4hid),color="#00688b")+
  geom_point(aes(x = MC_Samples,y = mean_error_4con_4hid),size = 1,shape = 21, fill = "blue")+
  scale_y_continuous(labels = scales::percent)+
  geom_line(aes(x = MC_Samples,y = mean_error_4con_4hid),color = "#0f0f43")+
  geom_smooth(aes(x = MC_Samples,y = mean_error_4con_4hid),size = 0.1,color="#8906d2",lty="dotted")+
  labs(x=" Number of MC Samples",
       y="Error Percentage")+
  theme_bw()




# 5 con  1 hid
mc_evalu_5con_1hid_1t_error <- 1-mc_evalu_5con_1hid_1t[,2]
mc_evalu_5con_1hid_2n_error <- 1-mc_evalu_5con_1hid_2n[,2]
mc_evalu_5con_1hid_3r_error <- 1-mc_evalu_5con_1hid_3r[,2]
mc_evalu_5con_1hid_4t_error <- 1-mc_evalu_5con_1hid_4t[,2]
mc_evalu_5con_1hid_5t_error <- 1-mc_evalu_5con_1hid_5t[,2]
mc_evalu_5con_1hid_6t_error <- 1-mc_evalu_5con_1hid_6t[,2]
mc_evalu_5con_1hid_7t_error <- 1-mc_evalu_5con_1hid_7t[,2]
mc_evalu_5con_1hid_8t_error <- 1-mc_evalu_5con_1hid_8t[,2]
mc_evalu_5con_1hid_9t_error <- 1-mc_evalu_5con_1hid_9t[,2]
mc_evalu_5con_1hid_10t_error <- 1-mc_evalu_5con_1hid_10t[,2]

all_error_5con_1hid_data <- as.data.frame(cbind(MC_Samples,mc_evalu_5con_1hid_1t_error,
                                                mc_evalu_5con_1hid_2n_error,
                                                mc_evalu_5con_1hid_3r_error,
                                                mc_evalu_5con_1hid_4t_error,
                                                mc_evalu_5con_1hid_5t_error,
                                                mc_evalu_5con_1hid_6t_error,
                                                mc_evalu_5con_1hid_7t_error,
                                                mc_evalu_5con_1hid_8t_error,
                                                mc_evalu_5con_1hid_9t_error,
                                                mc_evalu_5con_1hid_10t_error))


mean_error_5con_1hid <- apply(all_error_5con_1hid_data[,-1],1,mean) #50 output
sd_error_5con_1hid <- apply(all_error_5con_1hid_data[,-1],1,sd) #50 output
upper_5con_1hid <- mean_error_5con_1hid + sd_error_5con_1hid
lower_5con_1hid <- mean_error_5con_1hid - sd_error_5con_1hid



ggplot(data=all_error_5con_1hid_data)+
  geom_errorbar(aes(x = MC_Samples,ymin = lower_5con_1hid ,ymax = upper_5con_1hid),color="#00688b")+
  geom_point(aes(x = MC_Samples,y = mean_error_5con_1hid),size = 1,shape = 21, fill = "blue")+
  scale_y_continuous(labels = scales::percent)+
  geom_line(aes(x = MC_Samples,y = mean_error_5con_1hid),color = "#0f0f43")+
  geom_smooth(aes(x = MC_Samples,y = mean_error_5con_1hid),size = 0.1,color="#8906d2",lty="dotted")+
  labs(x=" Number of MC Samples",
       y="Error Percentage")+
  theme_bw()





# 5 con 2 hid
mc_evalu_5con_2hid_1t_error <- 1-mc_evalu_5con_2hid_1t[,2]
mc_evalu_5con_2hid_2n_error <- 1-mc_evalu_5con_2hid_2n[,2]
mc_evalu_5con_2hid_3r_error <- 1-mc_evalu_5con_2hid_3r[,2]
mc_evalu_5con_2hid_4t_error <- 1-mc_evalu_5con_2hid_4t[,2]
mc_evalu_5con_2hid_5t_error <- 1-mc_evalu_5con_2hid_5t[,2]
mc_evalu_5con_2hid_6t_error <- 1-mc_evalu_5con_2hid_6t[,2]
mc_evalu_5con_2hid_7t_error <- 1-mc_evalu_5con_2hid_7t[,2]
mc_evalu_5con_2hid_8t_error <- 1-mc_evalu_5con_2hid_8t[,2]
mc_evalu_5con_2hid_9t_error <- 1-mc_evalu_5con_2hid_9t[,2]
mc_evalu_5con_2hid_10t_error <- 1-mc_evalu_5con_2hid_10t[,2]

all_error_5con_2hid_data <- as.data.frame(cbind(MC_Samples,mc_evalu_5con_2hid_1t_error,
                                                mc_evalu_5con_2hid_2n_error,
                                                mc_evalu_5con_2hid_3r_error,
                                                mc_evalu_5con_2hid_4t_error,
                                                mc_evalu_5con_2hid_5t_error,
                                                mc_evalu_5con_2hid_6t_error,
                                                mc_evalu_5con_2hid_7t_error,
                                                mc_evalu_5con_2hid_8t_error,
                                                mc_evalu_5con_2hid_9t_error,
                                                mc_evalu_5con_2hid_10t_error))



mean_error_5con_2hid <- apply(all_error_5con_2hid_data[,-1],1,mean) #50 output
sd_error_5con_2hid <- apply(all_error_5con_2hid_data[,-1],1,sd) #50 output
upper_5con_2hid <- mean_error_5con_2hid + sd_error_5con_2hid 
lower_5con_2hid <- mean_error_5con_2hid - sd_error_5con_2hid 



ggplot(data=all_error_5con_2hid_data)+
  geom_errorbar(aes(x = MC_Samples,ymin = lower_5con_2hid ,ymax = upper_5con_2hid),color="#00688b")+
  geom_point(aes(x = MC_Samples,y = mean_error_5con_2hid),size = 1,shape = 21, fill = "blue")+
  scale_y_continuous(labels = scales::percent)+
  geom_line(aes(x = MC_Samples,y = mean_error_5con_2hid),color = "#0f0f43")+
  geom_smooth(aes(x = MC_Samples,y = mean_error_5con_2hid),size = 0.1,color="#8906d2",lty="dotted")+
  labs(x=" Number of MC Samples",
       y="Error Percentage")+
  theme_bw()




####################################################
##### Predictive uncertainty for best 2 networks
####################################################


#############################################
# 3con_5hid predictive uncertainty 
#############################################

# 1st model 

apply(mc_pred_3con_5hid_1t,1,mean)
apply(mc_pred_3con_5hid_1t,1,sd)

image <- 1:30
mc_pred_3con_5hid_1t_viz <- as.data.frame(cbind(image,mc_pred_3con_5hid_1t))
mc_pred_3con_5hid_1t_viz$mean_prob <- apply(mc_pred_3con_5hid_1t_viz[,-1],1,mean)
mc_pred_3con_5hid_1t_viz$sd_prob <- apply(mc_pred_3con_5hid_1t_viz[,2:51],1,sd)
mc_pred_3con_5hid_1t_viz$upper <- mc_pred_3con_5hid_1t_viz$mean_prob+mc_pred_3con_5hid_1t_viz$sd_prob
#mc_pred_3con_5hid_1t_viz$upper<- apply(mc_pred_3con_5hid_1t_viz[,2:51],1,max)
mc_pred_3con_5hid_1t_viz$lower <- mc_pred_3con_5hid_1t_viz$mean_prob-mc_pred_3con_5hid_1t_viz$sd_prob
#mc_pred_3con_5hid_1t_viz$lower<- apply(mc_pred_3con_5hid_1t_viz[,2:51],1,min)
mc_pred_3con_5hid_1t_viz$y_actual <- c(seq(0,0,length.out = 15),seq(1,1,length.out = 15))


ggplot(data=mc_pred_3con_5hid_1t_viz)+
  geom_errorbar(aes(x = image,ymin = lower ,ymax = upper),color="#703080",size=0.5)+
  geom_point(aes(x = image,y = mean_prob,fill=factor(y_actual)),
             size = 1.5,shape = 21)+ 
  scale_fill_discrete(name="Y Actual",labels=c("0 = No", "1 = Yes"))+
  labs(x=" Test Images",
       y="Mean Prediction Probabilities")+
  theme_bw()+
  theme(legend.justification=c(1,0), legend.position=c(1,0))+
  geom_rect(fill = 'red', xmin = -Inf, xmax = 15.5, ymin =0, ymax = 0.5, alpha =0.005)+
  geom_rect(fill = 'green', xmin = 15.5, xmax = Inf, ymin =0.5, ymax = 1, alpha =0.005)


# 2nd model 

apply(mc_pred_3con_5hid_2n,1,mean)
apply(mc_pred_3con_5hid_2n,1,sd)

image <- 1:30
mc_pred_3con_5hid_2n_viz <- as.data.frame(cbind(image,mc_pred_3con_5hid_2n))
mc_pred_3con_5hid_2n_viz$mean_prob <- apply(mc_pred_3con_5hid_2n_viz[,-1],1,mean)
mc_pred_3con_5hid_2n_viz$sd_prob <- apply(mc_pred_3con_5hid_2n_viz[,2:51],1,sd)
mc_pred_3con_5hid_2n_viz$upper <- mc_pred_3con_5hid_2n_viz$mean_prob+mc_pred_3con_5hid_2n_viz$sd_prob
#mc_pred_3con_5hid_2n_viz$upper<- apply(mc_pred_3con_5hid_2n_viz[,2:51],1,max)
mc_pred_3con_5hid_2n_viz$lower <- mc_pred_3con_5hid_2n_viz$mean_prob-mc_pred_3con_5hid_2n_viz$sd_prob
#mc_pred_3con_5hid_2n_viz$lower<- apply(mc_pred_3con_5hid_2n_viz[,2:51],1,min)
mc_pred_3con_5hid_2n_viz$y_actual <- c(seq(0,0,length.out = 15),seq(1,1,length.out = 15))


ggplot(data=mc_pred_3con_5hid_2n_viz)+
  geom_errorbar(aes(x = image,ymin = lower ,ymax = upper),color="#703080",size=0.5)+
  geom_point(aes(x = image,y = mean_prob,fill=factor(y_actual)),
             size = 1.5,shape = 21)+ 
  scale_fill_discrete(name="Y Actual",labels=c("0 = No", "1 = Yes"))+
  labs(x=" Test Images",
       y="Mean Prediction Probabilities")+
  theme_bw()+
  theme(legend.justification=c(0,1), legend.position=c(0,1))+
  geom_rect(fill = 'red', xmin = -Inf, xmax = 15.5, ymin =0, ymax = 0.5, alpha =0.005)+
  geom_rect(fill = 'green', xmin = 15.5, xmax = Inf, ymin =0.5, ymax = 1, alpha =0.005)


# 3rd model 

apply(mc_pred_3con_5hid_3r,1,mean)
apply(mc_pred_3con_5hid_3r,1,sd)

image <- 1:30
mc_pred_3con_5hid_3r_viz <- as.data.frame(cbind(image,mc_pred_3con_5hid_3r))
mc_pred_3con_5hid_3r_viz$mean_prob <- apply(mc_pred_3con_5hid_3r_viz[,-1],1,mean)
mc_pred_3con_5hid_3r_viz$sd_prob <- apply(mc_pred_3con_5hid_3r_viz[,2:51],1,sd)
mc_pred_3con_5hid_3r_viz$upper <- mc_pred_3con_5hid_3r_viz$mean_prob+mc_pred_3con_5hid_3r_viz$sd_prob
#mc_pred_3con_5hid_3r_viz$upper<- apply(mc_pred_3con_5hid_3r_viz[,2:51],1,max)
mc_pred_3con_5hid_3r_viz$lower <- mc_pred_3con_5hid_3r_viz$mean_prob-mc_pred_3con_5hid_3r_viz$sd_prob
#mc_pred_3con_5hid_3r_viz$lower<- apply(mc_pred_3con_5hid_3r_viz[,2:51],1,min)
mc_pred_3con_5hid_3r_viz$y_actual <- c(seq(0,0,length.out = 15),seq(1,1,length.out = 15))
sort(mc_pred_3con_5hid_3r_viz$mean_prob)

ggplot(data=mc_pred_3con_5hid_3r_viz)+
  geom_errorbar(aes(x = image,ymin = lower ,ymax = upper),color="#703080",size=0.5)+
  geom_point(aes(x = image,y = mean_prob,fill=factor(y_actual)),
             size = 1.5,shape = 21)+ 
  scale_fill_discrete(name="Y Actual",labels=c("0 = No", "1 = Yes"))+
  labs(x=" Test Images",
       y="Mean Prediction Probabilities")+
  theme_bw()+
  theme(legend.justification=c(1,0), legend.position=c(1,0))+
  geom_rect(fill = 'red', xmin = -Inf, xmax = 15.5, ymin =0, ymax = 0.5, alpha =0.005)+
  geom_rect(fill = 'green', xmin = 15.5, xmax = Inf, ymin =0.5, ymax = 1, alpha =0.005)


# 4th model 

apply(mc_pred_3con_5hid_4t,1,mean)
apply(mc_pred_3con_5hid_4t,1,sd)

image <- 1:30
mc_pred_3con_5hid_4t_viz <- as.data.frame(cbind(image,mc_pred_3con_5hid_4t))
mc_pred_3con_5hid_4t_viz$mean_prob <- apply(mc_pred_3con_5hid_4t_viz[,-1],1,mean)
mc_pred_3con_5hid_4t_viz$sd_prob <- apply(mc_pred_3con_5hid_4t_viz[,2:51],1,sd)
mc_pred_3con_5hid_4t_viz$upper <- mc_pred_3con_5hid_4t_viz$mean_prob+mc_pred_3con_5hid_4t_viz$sd_prob
#mc_pred_3con_5hid_4t_viz$upper<- apply(mc_pred_3con_5hid_4t_viz[,2:51],1,max)
mc_pred_3con_5hid_4t_viz$lower <- mc_pred_3con_5hid_4t_viz$mean_prob-mc_pred_3con_5hid_4t_viz$sd_prob
#mc_pred_3con_5hid_4t_viz$lower<- apply(mc_pred_3con_5hid_4t_viz[,2:51],1,min)
mc_pred_3con_5hid_4t_viz$y_actual <- c(seq(0,0,length.out = 15),seq(1,1,length.out = 15))


ggplot(data=mc_pred_3con_5hid_4t_viz)+
  geom_errorbar(aes(x = image,ymin = lower ,ymax = upper),color="#703080",size=0.5)+
  geom_point(aes(x = image,y = mean_prob,fill=factor(y_actual)),
             size = 1.5,shape = 21)+ 
  scale_fill_discrete(name="Y Actual",labels=c("0 = No", "1 = Yes"))+
  labs(x=" Test Images",
       y="Mean Prediction Probabilities")+
  theme_bw()+
  theme(legend.justification=c(1,0), legend.position=c(1,0))+
  geom_rect(fill = 'red', xmin = -Inf, xmax = 15.5, ymin =0, ymax = 0.5, alpha =0.005)+
  geom_rect(fill = 'green', xmin = 15.5, xmax = Inf, ymin =0.5, ymax = 1, alpha =0.005)



# 5th model 
apply(mc_pred_3con_5hid_5t,1,mean)
apply(mc_pred_3con_5hid_5t,1,sd)

image <- 1:30
mc_pred_3con_5hid_5t_viz <- as.data.frame(cbind(image,mc_pred_3con_5hid_5t))
mc_pred_3con_5hid_5t_viz$mean_prob <- apply(mc_pred_3con_5hid_5t_viz[,-1],1,mean)
mc_pred_3con_5hid_5t_viz$sd_prob <- apply(mc_pred_3con_5hid_5t_viz[,2:51],1,sd)
mc_pred_3con_5hid_5t_viz$upper <- mc_pred_3con_5hid_5t_viz$mean_prob+mc_pred_3con_5hid_5t_viz$sd_prob
#mc_pred_3con_5hid_5t_viz$upper<- apply(mc_pred_3con_5hid_5t_viz[,2:51],1,max)
mc_pred_3con_5hid_5t_viz$lower <- mc_pred_3con_5hid_5t_viz$mean_prob-mc_pred_3con_5hid_5t_viz$sd_prob
#mc_pred_3con_5hid_5t_viz$lower<- apply(mc_pred_3con_5hid_5t_viz[,2:51],1,min)
mc_pred_3con_5hid_5t_viz$y_actual <- c(seq(0,0,length.out = 15),seq(1,1,length.out = 15))


ggplot(data=mc_pred_3con_5hid_5t_viz)+
  geom_errorbar(aes(x = image,ymin = lower ,ymax = upper),color="#703080",size=0.5)+
  geom_point(aes(x = image,y = mean_prob,fill=factor(y_actual)),
             size = 1.5,shape = 21)+ 
  scale_fill_discrete(name="Y Actual",labels=c("0 = No", "1 = Yes"))+
  labs(x=" Test Images",
       y="Mean Prediction Probabilities")+
  theme_bw()+
  theme(legend.justification=c(1,0), legend.position=c(1,0))+
  geom_rect(fill = 'red', xmin = -Inf, xmax = 15.5, ymin =0, ymax = 0.5, alpha =0.005)+
  geom_rect(fill = 'green', xmin = 15.5, xmax = Inf, ymin =0.5, ymax = 1, alpha =0.005)


# 6th model 

apply(mc_pred_3con_5hid_6t,1,mean)
apply(mc_pred_3con_5hid_6t,1,sd)

image <- 1:30
mc_pred_3con_5hid_6t_viz <- as.data.frame(cbind(image,mc_pred_3con_5hid_6t))
mc_pred_3con_5hid_6t_viz$mean_prob <- apply(mc_pred_3con_5hid_6t_viz[,-1],1,mean)
mc_pred_3con_5hid_6t_viz$sd_prob <- apply(mc_pred_3con_5hid_6t_viz[,2:51],1,sd)
mc_pred_3con_5hid_6t_viz$upper <- mc_pred_3con_5hid_6t_viz$mean_prob+mc_pred_3con_5hid_6t_viz$sd_prob
#mc_pred_3con_5hid_6t_viz$upper<- apply(mc_pred_3con_5hid_6t_viz[,2:51],1,max)
mc_pred_3con_5hid_6t_viz$lower <- mc_pred_3con_5hid_6t_viz$mean_prob-mc_pred_3con_5hid_6t_viz$sd_prob
#mc_pred_3con_5hid_6t_viz$lower<- apply(mc_pred_3con_5hid_6t_viz[,2:51],1,min)
mc_pred_3con_5hid_6t_viz$y_actual <- c(seq(0,0,length.out = 15),seq(1,1,length.out = 15))


ggplot(data=mc_pred_3con_5hid_6t_viz)+
  geom_errorbar(aes(x = image,ymin = lower ,ymax = upper),color="#703080",size=0.5)+
  geom_point(aes(x = image,y = mean_prob,fill=factor(y_actual)),
             size = 1.5,shape = 21)+ 
  scale_fill_discrete(name="Y Actual",labels=c("0 = No", "1 = Yes"))+
  labs(x=" Test Images",
       y="Mean Prediction Probabilities")+
  theme_bw()+
  theme(legend.justification=c(1,0), legend.position=c(1,0))+
  geom_rect(fill = 'red', xmin = -Inf, xmax = 15.5, ymin =0, ymax = 0.5, alpha =0.005)+
  geom_rect(fill = 'green', xmin = 15.5, xmax = Inf, ymin =0.5, ymax = 1, alpha =0.005)



# 7th model 

apply(mc_pred_3con_5hid_7t,1,mean)
apply(mc_pred_3con_5hid_7t,1,sd)

image <- 1:30
mc_pred_3con_5hid_7t_viz <- as.data.frame(cbind(image,mc_pred_3con_5hid_7t))
mc_pred_3con_5hid_7t_viz$mean_prob <- apply(mc_pred_3con_5hid_7t_viz[,-1],1,mean)
mc_pred_3con_5hid_7t_viz$sd_prob <- apply(mc_pred_3con_5hid_7t_viz[,2:51],1,sd)
mc_pred_3con_5hid_7t_viz$upper <- mc_pred_3con_5hid_7t_viz$mean_prob+mc_pred_3con_5hid_7t_viz$sd_prob
#mc_pred_3con_5hid_7t_viz$upper<- apply(mc_pred_3con_5hid_7t_viz[,2:51],1,max)
mc_pred_3con_5hid_7t_viz$lower <- mc_pred_3con_5hid_7t_viz$mean_prob-mc_pred_3con_5hid_7t_viz$sd_prob
#mc_pred_3con_5hid_7t_viz$lower<- apply(mc_pred_3con_5hid_7t_viz[,2:51],1,min)
mc_pred_3con_5hid_7t_viz$y_actual <- c(seq(0,0,length.out = 15),seq(1,1,length.out = 15))


ggplot(data=mc_pred_3con_5hid_7t_viz)+
  geom_errorbar(aes(x = image,ymin = lower ,ymax = upper),color="#703080",size=0.5)+
  geom_point(aes(x = image,y = mean_prob,fill=factor(y_actual)),
             size = 1.5,shape = 21)+ 
  scale_fill_discrete(name="Y Actual",labels=c("0 = No", "1 = Yes"))+
  labs(x=" Test Images",
       y="Mean Prediction Probabilities")+
  theme_bw()+
  theme(legend.justification=c(1,0), legend.position=c(1,0))+
  geom_rect(fill = 'red', xmin = -Inf, xmax = 15.5, ymin =0, ymax = 0.5, alpha =0.005)+
  geom_rect(fill = 'green', xmin = 15.5, xmax = Inf, ymin =0.5, ymax = 1, alpha =0.005)


# 8th model 

apply(mc_pred_3con_5hid_8t,1,mean)
apply(mc_pred_3con_5hid_8t,1,sd)

image <- 1:30
mc_pred_3con_5hid_8t_viz <- as.data.frame(cbind(image,mc_pred_3con_5hid_8t))
mc_pred_3con_5hid_8t_viz$mean_prob <- apply(mc_pred_3con_5hid_8t_viz[,-1],1,mean)
mc_pred_3con_5hid_8t_viz$sd_prob <- apply(mc_pred_3con_5hid_8t_viz[,2:51],1,sd)
mc_pred_3con_5hid_8t_viz$upper <- mc_pred_3con_5hid_8t_viz$mean_prob+mc_pred_3con_5hid_8t_viz$sd_prob
#mc_pred_3con_5hid_8t_viz$upper<- apply(mc_pred_3con_5hid_8t_viz[,2:51],1,max)
mc_pred_3con_5hid_8t_viz$lower <- mc_pred_3con_5hid_8t_viz$mean_prob-mc_pred_3con_5hid_8t_viz$sd_prob
#mc_pred_3con_5hid_8t_viz$lower<- apply(mc_pred_3con_5hid_8t_viz[,2:51],1,min)
mc_pred_3con_5hid_8t_viz$y_actual <- c(seq(0,0,length.out = 15),seq(1,1,length.out = 15))


ggplot(data=mc_pred_3con_5hid_8t_viz)+
  geom_errorbar(aes(x = image,ymin = lower ,ymax = upper),color="#703080",size=0.5)+
  geom_point(aes(x = image,y = mean_prob,fill=factor(y_actual)),
             size = 1.5,shape = 21)+ 
  scale_fill_discrete(name="Y Actual",labels=c("0 = No", "1 = Yes"))+
  labs(x=" Test Images",
       y="Mean Prediction Probabilities")+
  theme_bw()+
  theme(legend.justification=c(1,0), legend.position=c(1,0))+
  geom_rect(fill = 'red', xmin = -Inf, xmax = 15.5, ymin =0, ymax = 0.5, alpha =0.005)+
  geom_rect(fill = 'green', xmin = 15.5, xmax = Inf, ymin =0.5, ymax = 1, alpha =0.005)


# 9th model 

apply(mc_pred_3con_5hid_9t,1,mean)
apply(mc_pred_3con_5hid_9t,1,sd)

image <- 1:30
mc_pred_3con_5hid_9t_viz <- as.data.frame(cbind(image,mc_pred_3con_5hid_9t))
mc_pred_3con_5hid_9t_viz$mean_prob <- apply(mc_pred_3con_5hid_9t_viz[,-1],1,mean)
mc_pred_3con_5hid_9t_viz$sd_prob <- apply(mc_pred_3con_5hid_9t_viz[,2:51],1,sd)
mc_pred_3con_5hid_9t_viz$upper <- mc_pred_3con_5hid_9t_viz$mean_prob+mc_pred_3con_5hid_9t_viz$sd_prob
#mc_pred_3con_5hid_9t_viz$upper<- apply(mc_pred_3con_5hid_9t_viz[,2:51],1,max)
mc_pred_3con_5hid_9t_viz$lower <- mc_pred_3con_5hid_9t_viz$mean_prob-mc_pred_3con_5hid_9t_viz$sd_prob
#mc_pred_3con_5hid_9t_viz$lower<- apply(mc_pred_3con_5hid_9t_viz[,2:51],1,min)
mc_pred_3con_5hid_9t_viz$y_actual <- c(seq(0,0,length.out = 15),seq(1,1,length.out = 15))


ggplot(data=mc_pred_3con_5hid_9t_viz)+
  geom_errorbar(aes(x = image,ymin = lower ,ymax = upper),color="#703080",size=0.5)+
  geom_point(aes(x = image,y = mean_prob,fill=factor(y_actual)),
             size = 1.5,shape = 21)+ 
  scale_fill_discrete(name="Y Actual",labels=c("0 = No", "1 = Yes"))+
  labs(x=" Test Images",
       y="Mean Prediction Probabilities")+
  theme_bw()+
  theme(legend.justification=c(1,0), legend.position=c(1,0))+
  geom_rect(fill = 'red', xmin = -Inf, xmax = 15.5, ymin =0, ymax = 0.5, alpha =0.005)+
  geom_rect(fill = 'green', xmin = 15.5, xmax = Inf, ymin =0.5, ymax = 1, alpha =0.005)



# 10th model (lworst error rate)

apply(mc_pred_3con_5hid_10t,1,mean)
apply(mc_pred_3con_5hid_10t,1,sd)

image <- 1:30
mc_pred_3con_5hid_10t_viz <- as.data.frame(cbind(image,mc_pred_3con_5hid_10t))
mc_pred_3con_5hid_10t_viz$mean_prob <- apply(mc_pred_3con_5hid_10t_viz[,-1],1,mean)
mc_pred_3con_5hid_10t_viz$sd_prob <- apply(mc_pred_3con_5hid_10t_viz[,2:51],1,sd)
mc_pred_3con_5hid_10t_viz$upper <- mc_pred_3con_5hid_10t_viz$mean_prob+mc_pred_3con_5hid_10t_viz$sd_prob
#mc_pred_3con_5hid_10t_viz$upper<- apply(mc_pred_3con_5hid_10t_viz[,2:51],1,max)
mc_pred_3con_5hid_10t_viz$lower <- mc_pred_3con_5hid_10t_viz$mean_prob-mc_pred_3con_5hid_10t_viz$sd_prob
#mc_pred_3con_5hid_10t_viz$lower<- apply(mc_pred_3con_5hid_10t_viz[,2:51],1,min)
mc_pred_3con_5hid_10t_viz$y_actual <- c(seq(0,0,length.out = 15),seq(1,1,length.out = 15))


ggplot(data=mc_pred_3con_5hid_10t_viz)+
  geom_errorbar(aes(x = image,ymin = lower ,ymax = upper),color="#703080",size=0.5)+
  geom_point(aes(x = image,y = mean_prob,fill=factor(y_actual)),
             size = 1.5,shape = 21)+ 
  scale_fill_discrete(name="Y Actual",labels=c("0 = No", "1 = Yes"))+
  labs(x=" Test Images",
       y="Mean Prediction Probabilities")+
  theme_bw()+
  theme(legend.justification=c(1,0), legend.position=c(1,0))+
  geom_rect(fill = 'red', xmin = -Inf, xmax = 15.5, ymin =0, ymax = 0.5, alpha =0.005)+
  geom_rect(fill = 'green', xmin = 15.5, xmax = Inf, ymin =0.5, ymax = 1, alpha =0.005)





#############################################
# 4con_5hid predictive uncertainty 
#############################################

# 1st model 

apply(mc_pred_4con_5hid_1t,1,mean)
apply(mc_pred_4con_5hid_1t,1,sd)

image <- 1:30
mc_pred_4con_5hid_1t_viz <- as.data.frame(cbind(image,mc_pred_4con_5hid_1t))
mc_pred_4con_5hid_1t_viz$mean_prob <- apply(mc_pred_4con_5hid_1t_viz[,-1],1,mean)
mc_pred_4con_5hid_1t_viz$sd_prob <- apply(mc_pred_4con_5hid_1t_viz[,2:51],1,sd)
mc_pred_4con_5hid_1t_viz$upper <- mc_pred_4con_5hid_1t_viz$mean_prob+mc_pred_4con_5hid_1t_viz$sd_prob
#mc_pred_4con_5hid_1t_viz$upper<- apply(mc_pred_4con_5hid_1t_viz[,2:51],1,max)
mc_pred_4con_5hid_1t_viz$lower <- mc_pred_4con_5hid_1t_viz$mean_prob-mc_pred_4con_5hid_1t_viz$sd_prob
#mc_pred_4con_5hid_1t_viz$lower<- apply(mc_pred_4con_5hid_1t_viz[,2:51],1,min)
mc_pred_4con_5hid_1t_viz$y_actual <- c(seq(0,0,length.out = 15),seq(1,1,length.out = 15))


ggplot(data=mc_pred_4con_5hid_1t_viz)+
  geom_errorbar(aes(x = image,ymin = lower ,ymax = upper),color="#703080",size=0.5)+
  geom_point(aes(x = image,y = mean_prob,fill=factor(y_actual)),
             size = 1.5,shape = 21)+ 
  scale_fill_discrete(name="Y Actual",labels=c("0 = No", "1 = Yes"))+
  labs(x=" Test Images",
       y="Mean Prediction Probabilities")+
  theme_bw()+
  theme(legend.justification=c(1,0), legend.position=c(1,0))+
  geom_rect(fill = 'red', xmin = -Inf, xmax = 15.5, ymin =0, ymax = 0.5, alpha =0.005)+
  geom_rect(fill = 'green', xmin = 15.5, xmax = Inf, ymin =0.5, ymax = 1, alpha =0.005)


# 2nd model 

apply(mc_pred_4con_5hid_2n,1,mean)
apply(mc_pred_4con_5hid_2n,1,sd)

image <- 1:30
mc_pred_4con_5hid_2n_viz <- as.data.frame(cbind(image,mc_pred_4con_5hid_2n))
mc_pred_4con_5hid_2n_viz$mean_prob <- apply(mc_pred_4con_5hid_2n_viz[,-1],1,mean)
mc_pred_4con_5hid_2n_viz$sd_prob <- apply(mc_pred_4con_5hid_2n_viz[,2:51],1,sd)
mc_pred_4con_5hid_2n_viz$upper <- mc_pred_4con_5hid_2n_viz$mean_prob+mc_pred_4con_5hid_2n_viz$sd_prob
#mc_pred_4con_5hid_2n_viz$upper<- apply(mc_pred_4con_5hid_2n_viz[,2:51],1,max)
mc_pred_4con_5hid_2n_viz$lower <- mc_pred_4con_5hid_2n_viz$mean_prob-mc_pred_4con_5hid_2n_viz$sd_prob
#mc_pred_4con_5hid_2n_viz$lower<- apply(mc_pred_4con_5hid_2n_viz[,2:51],1,min)
mc_pred_4con_5hid_2n_viz$y_actual <- c(seq(0,0,length.out = 15),seq(1,1,length.out = 15))


ggplot(data=mc_pred_4con_5hid_2n_viz)+
  geom_errorbar(aes(x = image,ymin = lower ,ymax = upper),color="#703080",size=0.5)+
  geom_point(aes(x = image,y = mean_prob,fill=factor(y_actual)),
             size = 1.5,shape = 21)+ 
  scale_fill_discrete(name="Y Actual",labels=c("0 = No", "1 = Yes"))+
  labs(x=" Test Images",
       y="Mean Prediction Probabilities")+
  theme_bw()+
  theme(legend.justification=c(0,1), legend.position=c(0,1))+
  geom_rect(fill = 'red', xmin = -Inf, xmax = 15.5, ymin =0, ymax = 0.5, alpha =0.005)+
  geom_rect(fill = 'green', xmin = 15.5, xmax = Inf, ymin =0.5, ymax = 1, alpha =0.005)


# 3rd model 

apply(mc_pred_4con_5hid_3r,1,mean)
apply(mc_pred_4con_5hid_3r,1,sd)

image <- 1:30
mc_pred_4con_5hid_3r_viz <- as.data.frame(cbind(image,mc_pred_4con_5hid_3r))
mc_pred_4con_5hid_3r_viz$mean_prob <- apply(mc_pred_4con_5hid_3r_viz[,-1],1,mean)
mc_pred_4con_5hid_3r_viz$sd_prob <- apply(mc_pred_4con_5hid_3r_viz[,2:51],1,sd)
mc_pred_4con_5hid_3r_viz$upper <- mc_pred_4con_5hid_3r_viz$mean_prob+mc_pred_4con_5hid_3r_viz$sd_prob
#mc_pred_4con_5hid_3r_viz$upper<- apply(mc_pred_4con_5hid_3r_viz[,2:51],1,max)
mc_pred_4con_5hid_3r_viz$lower <- mc_pred_4con_5hid_3r_viz$mean_prob-mc_pred_4con_5hid_3r_viz$sd_prob
#mc_pred_4con_5hid_3r_viz$lower<- apply(mc_pred_4con_5hid_3r_viz[,2:51],1,min)
mc_pred_4con_5hid_3r_viz$y_actual <- c(seq(0,0,length.out = 15),seq(1,1,length.out = 15))
sort(mc_pred_4con_5hid_3r_viz$mean_prob)

ggplot(data=mc_pred_4con_5hid_3r_viz)+
  geom_errorbar(aes(x = image,ymin = lower ,ymax = upper),color="#703080",size=0.5)+
  geom_point(aes(x = image,y = mean_prob,fill=factor(y_actual)),
             size = 1.5,shape = 21)+ 
  scale_fill_discrete(name="Y Actual",labels=c("0 = No", "1 = Yes"))+
  labs(x=" Test Images",
       y="Mean Prediction Probabilities")+
  theme_bw()+
  theme(legend.justification=c(1,0), legend.position=c(1,0))+
  geom_rect(fill = 'red', xmin = -Inf, xmax = 15.5, ymin =0, ymax = 0.5, alpha =0.005)+
  geom_rect(fill = 'green', xmin = 15.5, xmax = Inf, ymin =0.5, ymax = 1, alpha =0.005)


# 4th model 

apply(mc_pred_4con_5hid_4t,1,mean)
apply(mc_pred_4con_5hid_4t,1,sd)

image <- 1:30
mc_pred_4con_5hid_4t_viz <- as.data.frame(cbind(image,mc_pred_4con_5hid_4t))
mc_pred_4con_5hid_4t_viz$mean_prob <- apply(mc_pred_4con_5hid_4t_viz[,-1],1,mean)
mc_pred_4con_5hid_4t_viz$sd_prob <- apply(mc_pred_4con_5hid_4t_viz[,2:51],1,sd)
mc_pred_4con_5hid_4t_viz$upper <- mc_pred_4con_5hid_4t_viz$mean_prob+mc_pred_4con_5hid_4t_viz$sd_prob
#mc_pred_4con_5hid_4t_viz$upper<- apply(mc_pred_4con_5hid_4t_viz[,2:51],1,max)
mc_pred_4con_5hid_4t_viz$lower <- mc_pred_4con_5hid_4t_viz$mean_prob-mc_pred_4con_5hid_4t_viz$sd_prob
#mc_pred_4con_5hid_4t_viz$lower<- apply(mc_pred_4con_5hid_4t_viz[,2:51],1,min)
mc_pred_4con_5hid_4t_viz$y_actual <- c(seq(0,0,length.out = 15),seq(1,1,length.out = 15))


ggplot(data=mc_pred_4con_5hid_4t_viz)+
  geom_errorbar(aes(x = image,ymin = lower ,ymax = upper),color="#703080",size=0.5)+
  geom_point(aes(x = image,y = mean_prob,fill=factor(y_actual)),
             size = 1.5,shape = 21)+ 
  scale_fill_discrete(name="Y Actual",labels=c("0 = No", "1 = Yes"))+
  labs(x=" Test Images",
       y="Mean Prediction Probabilities")+
  theme_bw()+
  theme(legend.justification=c(1,0), legend.position=c(1,0))+
  geom_rect(fill = 'red', xmin = -Inf, xmax = 15.5, ymin =0, ymax = 0.5, alpha =0.005)+
  geom_rect(fill = 'green', xmin = 15.5, xmax = Inf, ymin =0.5, ymax = 1, alpha =0.005)



# 5th model 
apply(mc_pred_4con_5hid_5t,1,mean)
apply(mc_pred_4con_5hid_5t,1,sd)

image <- 1:30
mc_pred_4con_5hid_5t_viz <- as.data.frame(cbind(image,mc_pred_4con_5hid_5t))
mc_pred_4con_5hid_5t_viz$mean_prob <- apply(mc_pred_4con_5hid_5t_viz[,-1],1,mean)
mc_pred_4con_5hid_5t_viz$sd_prob <- apply(mc_pred_4con_5hid_5t_viz[,2:51],1,sd)
mc_pred_4con_5hid_5t_viz$upper <- mc_pred_4con_5hid_5t_viz$mean_prob+mc_pred_4con_5hid_5t_viz$sd_prob
#mc_pred_4con_5hid_5t_viz$upper<- apply(mc_pred_4con_5hid_5t_viz[,2:51],1,max)
mc_pred_4con_5hid_5t_viz$lower <- mc_pred_4con_5hid_5t_viz$mean_prob-mc_pred_4con_5hid_5t_viz$sd_prob
#mc_pred_4con_5hid_5t_viz$lower<- apply(mc_pred_4con_5hid_5t_viz[,2:51],1,min)
mc_pred_4con_5hid_5t_viz$y_actual <- c(seq(0,0,length.out = 15),seq(1,1,length.out = 15))


ggplot(data=mc_pred_4con_5hid_5t_viz)+
  geom_errorbar(aes(x = image,ymin = lower ,ymax = upper),color="#703080",size=0.5)+
  geom_point(aes(x = image,y = mean_prob,fill=factor(y_actual)),
             size = 1.5,shape = 21)+ 
  scale_fill_discrete(name="Y Actual",labels=c("0 = No", "1 = Yes"))+
  labs(x=" Test Images",
       y="Mean Prediction Probabilities")+
  theme_bw()+
  theme(legend.justification=c(1,0), legend.position=c(1,0))+
  geom_rect(fill = 'red', xmin = -Inf, xmax = 15.5, ymin =0, ymax = 0.5, alpha =0.005)+
  geom_rect(fill = 'green', xmin = 15.5, xmax = Inf, ymin =0.5, ymax = 1, alpha =0.005)


# 6th model 

apply(mc_pred_4con_5hid_6t,1,mean)
apply(mc_pred_4con_5hid_6t,1,sd)

image <- 1:30
mc_pred_4con_5hid_6t_viz <- as.data.frame(cbind(image,mc_pred_4con_5hid_6t))
mc_pred_4con_5hid_6t_viz$mean_prob <- apply(mc_pred_4con_5hid_6t_viz[,-1],1,mean)
mc_pred_4con_5hid_6t_viz$sd_prob <- apply(mc_pred_4con_5hid_6t_viz[,2:51],1,sd)
mc_pred_4con_5hid_6t_viz$upper <- mc_pred_4con_5hid_6t_viz$mean_prob+mc_pred_4con_5hid_6t_viz$sd_prob
#mc_pred_4con_5hid_6t_viz$upper<- apply(mc_pred_4con_5hid_6t_viz[,2:51],1,max)
mc_pred_4con_5hid_6t_viz$lower <- mc_pred_4con_5hid_6t_viz$mean_prob-mc_pred_4con_5hid_6t_viz$sd_prob
#mc_pred_4con_5hid_6t_viz$lower<- apply(mc_pred_4con_5hid_6t_viz[,2:51],1,min)
mc_pred_4con_5hid_6t_viz$y_actual <- c(seq(0,0,length.out = 15),seq(1,1,length.out = 15))


ggplot(data=mc_pred_4con_5hid_6t_viz)+
  geom_errorbar(aes(x = image,ymin = lower ,ymax = upper),color="#703080",size=0.5)+
  geom_point(aes(x = image,y = mean_prob,fill=factor(y_actual)),
             size = 1.5,shape = 21)+ 
  scale_fill_discrete(name="Y Actual",labels=c("0 = No", "1 = Yes"))+
  labs(x=" Test Images",
       y="Mean Prediction Probabilities")+
  theme_bw()+
  theme(legend.justification=c(1,0), legend.position=c(1,0))+
  geom_rect(fill = 'red', xmin = -Inf, xmax = 15.5, ymin =0, ymax = 0.5, alpha =0.005)+
  geom_rect(fill = 'green', xmin = 15.5, xmax = Inf, ymin =0.5, ymax = 1, alpha =0.005)



# 7th model 

apply(mc_pred_4con_5hid_7t,1,mean)
apply(mc_pred_4con_5hid_7t,1,sd)

image <- 1:30
mc_pred_4con_5hid_7t_viz <- as.data.frame(cbind(image,mc_pred_4con_5hid_7t))
mc_pred_4con_5hid_7t_viz$mean_prob <- apply(mc_pred_4con_5hid_7t_viz[,-1],1,mean)
mc_pred_4con_5hid_7t_viz$sd_prob <- apply(mc_pred_4con_5hid_7t_viz[,2:51],1,sd)
mc_pred_4con_5hid_7t_viz$upper <- mc_pred_4con_5hid_7t_viz$mean_prob+mc_pred_4con_5hid_7t_viz$sd_prob
#mc_pred_4con_5hid_7t_viz$upper<- apply(mc_pred_4con_5hid_7t_viz[,2:51],1,max)
mc_pred_4con_5hid_7t_viz$lower <- mc_pred_4con_5hid_7t_viz$mean_prob-mc_pred_4con_5hid_7t_viz$sd_prob
#mc_pred_4con_5hid_7t_viz$lower<- apply(mc_pred_4con_5hid_7t_viz[,2:51],1,min)
mc_pred_4con_5hid_7t_viz$y_actual <- c(seq(0,0,length.out = 15),seq(1,1,length.out = 15))


ggplot(data=mc_pred_4con_5hid_7t_viz)+
  geom_errorbar(aes(x = image,ymin = lower ,ymax = upper),color="#703080",size=0.5)+
  geom_point(aes(x = image,y = mean_prob,fill=factor(y_actual)),
             size = 1.5,shape = 21)+ 
  scale_fill_discrete(name="Y Actual",labels=c("0 = No", "1 = Yes"))+
  labs(x=" Test Images",
       y="Mean Prediction Probabilities")+
  theme_bw()+
  theme(legend.justification=c(1,0), legend.position=c(1,0))+
  geom_rect(fill = 'red', xmin = -Inf, xmax = 15.5, ymin =0, ymax = 0.5, alpha =0.005)+
  geom_rect(fill = 'green', xmin = 15.5, xmax = Inf, ymin =0.5, ymax = 1, alpha =0.005)


# 8th model 

apply(mc_pred_4con_5hid_8t,1,mean)
apply(mc_pred_4con_5hid_8t,1,sd)

image <- 1:30
mc_pred_4con_5hid_8t_viz <- as.data.frame(cbind(image,mc_pred_4con_5hid_8t))
mc_pred_4con_5hid_8t_viz$mean_prob <- apply(mc_pred_4con_5hid_8t_viz[,-1],1,mean)
mc_pred_4con_5hid_8t_viz$sd_prob <- apply(mc_pred_4con_5hid_8t_viz[,2:51],1,sd)
mc_pred_4con_5hid_8t_viz$upper <- mc_pred_4con_5hid_8t_viz$mean_prob+mc_pred_4con_5hid_8t_viz$sd_prob
#mc_pred_4con_5hid_8t_viz$upper<- apply(mc_pred_4con_5hid_8t_viz[,2:51],1,max)
mc_pred_4con_5hid_8t_viz$lower <- mc_pred_4con_5hid_8t_viz$mean_prob-mc_pred_4con_5hid_8t_viz$sd_prob
#mc_pred_4con_5hid_8t_viz$lower<- apply(mc_pred_4con_5hid_8t_viz[,2:51],1,min)
mc_pred_4con_5hid_8t_viz$y_actual <- c(seq(0,0,length.out = 15),seq(1,1,length.out = 15))


ggplot(data=mc_pred_4con_5hid_8t_viz)+
  geom_errorbar(aes(x = image,ymin = lower ,ymax = upper),color="#703080",size=0.5)+
  geom_point(aes(x = image,y = mean_prob,fill=factor(y_actual)),
             size = 1.5,shape = 21)+ 
  scale_fill_discrete(name="Y Actual",labels=c("0 = No", "1 = Yes"))+
  labs(x=" Test Images",
       y="Mean Prediction Probabilities")+
  theme_bw()+
  theme(legend.justification=c(1,0), legend.position=c(1,0))+
  geom_rect(fill = 'red', xmin = -Inf, xmax = 15.5, ymin =0, ymax = 0.5, alpha =0.005)+
  geom_rect(fill = 'green', xmin = 15.5, xmax = Inf, ymin =0.5, ymax = 1, alpha =0.005)


# 9th model 

apply(mc_pred_4con_5hid_9t,1,mean)
apply(mc_pred_4con_5hid_9t,1,sd)

image <- 1:30
mc_pred_4con_5hid_9t_viz <- as.data.frame(cbind(image,mc_pred_4con_5hid_9t))
mc_pred_4con_5hid_9t_viz$mean_prob <- apply(mc_pred_4con_5hid_9t_viz[,-1],1,mean)
mc_pred_4con_5hid_9t_viz$sd_prob <- apply(mc_pred_4con_5hid_9t_viz[,2:51],1,sd)
mc_pred_4con_5hid_9t_viz$upper <- mc_pred_4con_5hid_9t_viz$mean_prob+mc_pred_4con_5hid_9t_viz$sd_prob
#mc_pred_4con_5hid_9t_viz$upper<- apply(mc_pred_4con_5hid_9t_viz[,2:51],1,max)
mc_pred_4con_5hid_9t_viz$lower <- mc_pred_4con_5hid_9t_viz$mean_prob-mc_pred_4con_5hid_9t_viz$sd_prob
#mc_pred_4con_5hid_9t_viz$lower<- apply(mc_pred_4con_5hid_9t_viz[,2:51],1,min)
mc_pred_4con_5hid_9t_viz$y_actual <- c(seq(0,0,length.out = 15),seq(1,1,length.out = 15))


ggplot(data=mc_pred_4con_5hid_9t_viz)+
  geom_errorbar(aes(x = image,ymin = lower ,ymax = upper),color="#703080",size=0.5)+
  geom_point(aes(x = image,y = mean_prob,fill=factor(y_actual)),
             size = 1.5,shape = 21)+ 
  scale_fill_discrete(name="Y Actual",labels=c("0 = No", "1 = Yes"))+
  labs(x=" Test Images",
       y="Mean Prediction Probabilities")+
  theme_bw()+
  theme(legend.justification=c(1,0), legend.position=c(1,0))+
  geom_rect(fill = 'red', xmin = -Inf, xmax = 15.5, ymin =0, ymax = 0.5, alpha =0.005)+
  geom_rect(fill = 'green', xmin = 15.5, xmax = Inf, ymin =0.5, ymax = 1, alpha =0.005)



# 10th model (lworst error rate)

apply(mc_pred_4con_5hid_10t,1,mean)
apply(mc_pred_4con_5hid_10t,1,sd)

image <- 1:30
mc_pred_4con_5hid_10t_viz <- as.data.frame(cbind(image,mc_pred_4con_5hid_10t))
mc_pred_4con_5hid_10t_viz$mean_prob <- apply(mc_pred_4con_5hid_10t_viz[,-1],1,mean)
mc_pred_4con_5hid_10t_viz$sd_prob <- apply(mc_pred_4con_5hid_10t_viz[,2:51],1,sd)
mc_pred_4con_5hid_10t_viz$upper <- mc_pred_4con_5hid_10t_viz$mean_prob+mc_pred_4con_5hid_10t_viz$sd_prob
#mc_pred_4con_5hid_10t_viz$upper<- apply(mc_pred_4con_5hid_10t_viz[,2:51],1,max)
mc_pred_4con_5hid_10t_viz$lower <- mc_pred_4con_5hid_10t_viz$mean_prob-mc_pred_4con_5hid_10t_viz$sd_prob
#mc_pred_4con_5hid_10t_viz$lower<- apply(mc_pred_4con_5hid_10t_viz[,2:51],1,min)
mc_pred_4con_5hid_10t_viz$y_actual <- c(seq(0,0,length.out = 15),seq(1,1,length.out = 15))


ggplot(data=mc_pred_4con_5hid_10t_viz)+
  geom_errorbar(aes(x = image,ymin = lower ,ymax = upper),color="#703080",size=0.5)+
  geom_point(aes(x = image,y = mean_prob,fill=factor(y_actual)),
             size = 1.5,shape = 21)+ 
  scale_fill_discrete(name="Y Actual",labels=c("0 = No", "1 = Yes"))+
  labs(x=" Test Images",
       y="Mean Prediction Probabilities")+
  theme_bw()+
  theme(legend.justification=c(1,0), legend.position=c(1,0))+
  geom_rect(fill = 'red', xmin = -Inf, xmax = 15.5, ymin =0, ymax = 0.5, alpha =0.005)+
  geom_rect(fill = 'green', xmin = 15.5, xmax = Inf, ymin =0.5, ymax = 1, alpha =0.005)






#########################
## OVERFITTING
## Do it for all models
#########################

ppp <- cbind(subset(history_df_do_3c_5h_1t, data=="validation" & metric=="acc"),
             mc_evalu_3con_5hid_1t)


cols <- c("Validation"="#199999","Test"="#ff7373" )
ggplot(data=ppp)+
  #Evalidation
  geom_smooth(aes(x=epoch,y=value,colour="Validation"),size=0.8,alpha=0.8,show.legend = TRUE)+
  geom_point(aes(x=epoch,y=value),color="#199999",shape=17,alpha=0.7,size=1.5)+
  #test
  geom_smooth(aes(x=epoch,y=Accuracy,colour="Test"),size=0.8,alpha=0.8,show.legend = TRUE)+
  geom_point(aes(x=epoch,y=Accuracy),color="#941414",alpha=0.5,size=1.5)+
  labs(x=" Epochs / Iterations",
       y="Accuracy Rate")+
  scale_colour_manual(name="Accuracy", values=cols)+
  theme_bw()+
  theme(legend.justification=c(1,0), legend.position=c(1,0))



#############################################
#############################################





