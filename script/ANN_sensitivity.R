
#' ---
#' #' title: Example Script: A new method of ANN feature importance 
#' author: Tyler Gagne
#' date: Spring 2018
#' output: github_document
#' ---

# This is a draft script of a sensitivity analysis of variable importance
# We utilize the variable perturbation methodlogy as a means to test variable importance

# Under repeated ANN training rounds of weighting coefficients on random subsets of the training data
# we get a distribution of feature importance rankings that better approximate the stochastic range of
# feature importance approximation.

# load relevant libraries
library(ggridges)
library(mxnet)
library(caret)
library(dplyr)
library(reshape2)
library(rgdal)
library(broom)

##############################
###  Custom ggPlot theme   ###
##############################
themeo <-theme_classic()+
  theme(strip.background = element_blank(),
        axis.line = element_blank(),
        axis.text.x = element_text(margin = margin( 0.2, unit = "cm")),
        axis.text.y = element_text(margin = margin(c(1, 0.2), unit = "cm")),
        axis.ticks.length=unit(-0.1, "cm"),
        panel.border = element_rect(colour = "black", fill=NA, size=.5),
        legend.title=element_blank(),
        strip.text=element_text(hjust=0)
  )

#load data
load('../data/example.RData')
# Equal Area projection with Pacific Ocean centered meridian
countries_df <- readOGR(dsn = "../data/country_repro1", layer = "ocean_50m") %>% tidy() 

#########################
# Neural network build: #
#########################

# establish hyper parameters of MXnet ANN, these are drawn from an earlier training script, Keras version available
method_mod <- 'mxnet'
numFolds <- trainControl(method = 'none')   

tuneGrid <- expand.grid(
  layer1 = 10
  ,layer2 = 10
  ,layer3 = 10
  ,learning.rate = 0.02
  ,momentum = 0.9
  ,dropout = .2
  ,activation = "relu"
)

# This function takes a model object, pertubs each feature uniformily random and records the drop in 
# RMSE and R^2
Perturbed_Importance_sim <- 
  function(model_object,test_data,response_var,variable_names, scaled = F){
    
    # test set split fit metrics
    actual    <- test_data[response_var]
    predicted <- predict(model_object, test_data)
    #test_RMSE <- sqrt(mean((actual - predicted)^2))
    test_R2   <- 1 - (sum((actual-predicted)^2)/sum((actual-mean(actual[,1]))^2)) *100
    
    var_import <- NULL   
    for(x in 1:length(variable_names)){
      TRAIN_data <- test_data
      # shuffle variable
      TRAIN_data[,variable_names[x]] <- sample(TRAIN_data[,variable_names[x]]) # introduces some instability as well, perhaps set.seed.
      #predict with trained model
      var_import$pred_var[x] <- variable_names[x]
      # calculate test RMSE with noised variable
      #var_import$pred_var_RMSE[x] <- sqrt(mean((TRAIN_data[response_var] - predict(model_object, TRAIN_data))^2))
      var_import$pred_var_R2[x]   <- 1 - (sum((TRAIN_data[response_var]-predict(model_object, TRAIN_data))^2)/
                                            sum((TRAIN_data[response_var]-mean(TRAIN_data[response_var][,1]))^2)) *100
      
    }
    
    var_import <- as.data.frame(var_import)
    #var_import$pred_var_RMSE <- test_RMSE - var_import$pred_var_RMSE
    var_import$pred_var_R2   <- test_R2   - var_import$pred_var_R2
    
    if(scaled){ 
      var_import[,2:3] <- rangisize(var_import[,2:3])
    }
    
    print(var_import)
  }


sims <- 5  # this is a very small test simulation, suggested 500+ takes approximately 2 hours, quicker in paralell

# Conducted on both terrestrial and marine datasets of species biodiversity
# Within the loop below we conduct the input perturbation measure on ANN (MXnet or Keras)
# that were trained on 80% subsets of the training data. 
# Subsequently, were able to repeat that X amount of times. From there we establish a distribution
# of feature importance values under perturbation. Such that features that exhibit a narrow distribution
# are highly confined and regular in their effect on model performance, whereas features with high spread
# in feature importance are more highly variable in their predictive performance under inclusion.

# TERRESTRIAL
sim_runs <- data.frame(pred_vars = Land_varnames)

for(i in 1:sims){
  # sample to speed workup time
  LAND_noNA  <- LAND_submission #%>% sample_n(10000, replace = T)
  # split in to test and training sets
  # create data partition for training set 
  LandInTrain  <- createDataPartition(LAND_noNA$terrestrial_richness, p=0.8,list=FALSE)
  # partion out training set
  LAND_trainer<-LAND_noNA[LandInTrain,]
  # partition out test set
  LAND_test<-LAND_noNA[-LandInTrain,]
  
  # land tune
  terre <- train(y = LAND_trainer[,"terrestrial_richness"],
                 x = LAND_trainer[,Land_varnames],
                 method     = method_mod, 
                 preProcess = c('center', 'scale'), 
                 trControl  = numFolds, 
                 tuneGrid   = tuneGrid
  )
  # note the input, by inputting land trainer, its a train/leave 20% out
  # par(mfrow=c(6,1))
  var_importN <- Perturbed_Importance_sim(terre,LAND_trainer,"terrestrial_richness",Land_varnames,scaled = F)[[2]] 
  
  sim_runs[,i+1] <- var_importN
  print(i)
}

sim_joy <- melt(sim_runs)
prov_levels <- sim_joy %>%               # Reorder levels by mean importance of feature 
  dplyr::select(value,pred_vars) %>%
  dplyr::group_by(pred_vars) %>%
  dplyr::mutate(median_of_R2 = median(value))
prov_levels<- data.frame(prov_levels)
levels <- unique(prov_levels$pred_vars[order(prov_levels$median_of_R2)])

sim_joy$pred_vars<-factor(sim_joy$pred_vars, levels = levels, ordered = T)

str(LAND_trainer)

sim_joy_land <- sim_joy

ggplot(sim_joy_land, aes(x = value, y = pred_vars)) + 
  geom_density_ridges(
                      #rel_min_height = 0.03,
                      bandwidth = .1,
                      scale = 3,
                      alpha = .5,
                      color = "#1a9850",
                      fill  = "#1a9850")+
  scale_y_discrete(expand = c(0.01, 0)) +   # will generally have to set the `expand` option
  #scale_x_continuous(limits = c(NA, .5),expand = c(0, 0)) + 
  scale_fill_cyclical(values = c("light gray", "light gray"))+
  ylab(NULL)+xlab(NULL)+
  theme(axis.text.y = element_text(size=6),
        axis.text.x = element_text(size=6))+themeo

#ggplot(LAND_submission,aes(x,y,fill = sub_cv_pp))+geom_raster()+coord_fixed()


#### Repeat all of the above for the marine/ocean data ####


# MARINE
sim_runs <- data.frame(pred_vars = sea_varnames)

for(i in 1:sims){
  # sample to speed workup time
  OCEAN_noNA  <- SEA_submission #%>% sample_n(10000, replace = T)
  # split in to test and training sets
  # create data partition for training set 
  OceanInTrain  <- createDataPartition(OCEAN_noNA$marine_richness, p=0.8,list=FALSE)
  # partion out training set
  OCEAN_trainer<-OCEAN_noNA[OceanInTrain,]
  # partition out test set
  OCEAN_test<-OCEAN_noNA[-OceanInTrain,]
  
  # land tune
  oceaus <- train(y = OCEAN_trainer[,"marine_richness"],
                 x = OCEAN_trainer[,sea_varnames],
                 method     = method_mod, 
                 preProcess = c('center', 'scale'), 
                 trControl  = numFolds, 
                 tuneGrid   = tuneGrid
  )
  # note the input, by inputting ocean trainer, in a way its train/leave 20% out
  var_importN <- Perturbed_Importance_sim(oceaus,OCEAN_trainer,"marine_richness",sea_varnames,scaled = F)[[2]]
  
  sim_runs[,i+1] <- var_importN
  print(i)
}

sim_joy <- melt(sim_runs)
prov_levels <- sim_joy %>%               # Reorder levels by mean risk by privince 
  dplyr::select(value,pred_vars) %>%
  dplyr::group_by(pred_vars) %>%
  dplyr::mutate(median_of_R2 = median(value))
prov_levels<- data.frame(prov_levels)
levels <- unique(prov_levels$pred_vars[order(prov_levels$median_of_R2)])

sim_joy$pred_vars<-factor(sim_joy$pred_vars, levels = levels, ordered = T)

str(OCEAN_trainer)

sim_joy_marine <- sim_joy

ggplot(sim_joy_marine, aes(x = value, y = pred_vars)) + 
  geom_density_ridges(
    #rel_min_height = 0.03,
    bandwidth = .1,
    scale = 3,
    alpha = .5,
    color = "#4575b4",
    fill = "#4575b4")+
  scale_y_discrete(expand = c(0.01, 0)) +   # will generally have to set the `expand` option
  #scale_x_continuous(limits = c(NA, .5),expand = c(0, 0)) + 
  scale_fill_cyclical(values = c("light gray", "light gray"))+
  ylab(NULL)+xlab(NULL)+
  theme(axis.text.y = element_text(size=6),
        axis.text.x = element_text(size=6))+themeo

#ggplot(SEA_submission,aes(x,y,fill = ann_mean_sun))+geom_raster()+coord_fixed()



##



# run this post full model build in single_file_script

# select top X# of predictors and fit model

# build partial dependency grid
str(sim_joy_land)
prov_levels <- sim_joy_land %>%               # Reorder levels by mean risk by privince 
  dplyr::select(value,pred_vars) %>%
  dplyr::group_by(pred_vars) %>%
  dplyr::mutate(median_of_R2 = median(value))
prov_levels<- data.frame(prov_levels)
land_vars <- as.character(rev(unique(prov_levels$pred_vars[order(prov_levels$median_of_R2)])))[1:15]
land_vars # these are the top 15 predictors from above


str(sim_joy_marine)
prov_levels <- sim_joy_marine %>%               # Reorder levels by mean risk by privince 
  dplyr::select(value,pred_vars) %>%
  dplyr::group_by(pred_vars) %>%
  dplyr::mutate(median_of_R2 = median(value))
prov_levels<- data.frame(prov_levels)
sea_vars <- as.character(rev(unique(prov_levels$pred_vars[order(prov_levels$median_of_R2)])))[1:15]
sea_vars # these are the top 15 predictors from above

# subsetted model build

# build model on top 15 predictors
# land tune
numFolds <- trainControl(method = 'cv', number = 5, verboseIter = TRUE)  # establish training regim, 5-fold cross validation to monitor model stability

terre <- train(y = LAND_trainer[,"terrestrial_richness"],
               x = LAND_trainer[,land_vars],
               method     = method_mod, 
               preProcess = c('center', 'scale'), 
               trControl  = numFolds, 
               tuneGrid   = tuneGrid
               #,metric     = "rmse"
               #,epochs     = 50
)
terre

# sea tune
oceaus <-train(y = OCEAN_trainer[,"marine_richness"],
               x = OCEAN_trainer[,sea_vars],
               method     = method_mod, 
               preProcess = c('center', 'scale'), 
               trControl  = numFolds, 
               tuneGrid   = tuneGrid
               #,metric     = "rmse"
               #,epochs     = 50
)
oceaus

# plot residuals and compare to full model
LAND_submission$modeled <- predict(object = terre, newdata = LAND_submission)
LAND_submission$mod_error <- LAND_submission$terrestrial_richness - LAND_submission$modeled
SEA_submission$modeled <- predict(object = oceaus, newdata = SEA_submission)
SEA_submission$mod_error <- SEA_submission$marine_richness - SEA_submission$modeled

SEA_submission[,"mod_error"] <- scales::rescale(x = SEA_submission[,"mod_error"], to = c(0,1))
LAND_submission[,"mod_error"] <- scales::rescale(x = LAND_submission[,"mod_error"], to = c(0,1))


# overlayed, red means the model predicted more richness than observed, blue means the model under predicted observed richness
ggplot(SEA_submission)+
  geom_raster(data = SEA_submission, aes(x,y, fill = mod_error))+coord_fixed()+
  geom_raster(data = LAND_submission, aes(x,y, fill = mod_error))+coord_fixed()+
  geom_path(data = countries_df,aes(x= long, y = lat, group = group), color = "black", size = .25)+
  scale_fill_gradientn(colours = rev(c("#2166ac","#2166ac","#4575b4","#74add1","#d1e5f0","white","white","white","white","#fddbc7","#f46d43","#d73027","#b2182b","#b2182b")))+
  #scale_fill_distiller(palette = "RdBu", direction = 1)+
  
  themeo+
  scale_x_continuous(expand = c(-0.005,-0.005)) +
  scale_y_continuous(expand = c(-0.01,-0.01)) +
  #theme(axis.text = element_blank(),
  #    axis.title = element_blank())+
  ggtitle("A. model residuals w/o SAC covariate w error rescaled")+
  coord_fixed()


# build grid of pdep plots


# This is a brief loop to plot 2D partial dependency plots of the models developed above
#par(mfrow = c(5,6))
# centered on 0, this does take some time ~15 mins.

#for(i in 1:length(land_vars)) {
#  Land_varname <- land_vars[i]
#  pd_land <- partial(terre, pred.var = Land_varname)
#  pd_land$yhat <- pd_land$yhat - pd_land$yhat[1]
#  plot(pd_land[,1],pd_land[,2],type = 'l', xlab = colnames(pd_land)[1],ylab = 'tp', col = "#1a9850",ylim = c(-0.3,0.3))
  
#  sea_varname <- sea_vars[i]
#  pd_sea <- partial(oceaus, pred.var = sea_varname)
#  pd_sea$yhat <- pd_sea$yhat - pd_sea$yhat[1]
#  lines(pd_sea[,1],pd_sea[,2],type = 'l', xlab = colnames(pd_sea)[1],ylab = 'tp', col = "#4575b4")
#}








