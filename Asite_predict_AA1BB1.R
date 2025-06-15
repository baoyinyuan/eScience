#######################################################
# Predict HPC with trained model with XGBoost #
######################################################

# load library
packages <- c("readxl", "tidyr", "dplyr", "tidyverse", "magrittr","caret", "tictoc",
              "gridExtra","neuralnet","matrixStats","ggplot2", "ggExtra","gridExtra",
              "NeuralNetTools")
for(pkg in packages){
  library(pkg, character.only = TRUE)
}

# load source script and data
df_elem <- read.csv("elements_data.csv") %>% 
  as.data.frame()
# source("initial_descriptor_customize.R")

# Input experiment conditions
df_expe_user = as.data.frame(matrix(nrow = 1, ncol = 2))
colnames(df_expe_user) = c("temperature", "pH2O")
df_expe_user$temperature = 500  ## can be adjusted
df_expe_user$pH2O = 0.02


# Input compositions of AB_{1-x}B1_{x}O3
A_vector <- c("Ba", "Ca", "Sr", "Pr", "La", "Nd", "Sm", "Bi")
A_valence_vector <- c(2, 2, 2, 3, 3, 3, 3, 3)
# A_vector <- c("Ba", "Ca")
# A_valence_vector <- c(2, 2)
(A_fraction_vector <- seq(0.9, 0.5, by = - 0.2))
# A_fraction_vector <- 0.95

A1_vector <- c("Ba", "Ca", "Sr", "Pr", "La", "Nd", "Sm", "Bi")
A1_valence_vector <- c(2, 2, 2, 3, 3, 3, 3, 3)
# A1_vector <- c("Sr", "Pr", "La")
# A1_valence_vector <- c(2, 3, 3)
(A1_fraction_vector <- 1 - A_fraction_vector)

#
B_vector <- c("Al", "Mg","Si", "Ca", "Ti", "Mn", "Fe")
B_valence_vector <- c(3, 2, 4, 2, 4, 3, 3)
(B_fraction_vector <- seq(0.9, 0.7, by = - 0.1))
# (B_fraction_vector <- 1)

B1_vector <- c("Al", "Mg", "Si", "Ca", "Sc", "Mn", "Fe")
B1_valence_vector <- c(3, 2, 4, 2, 3, 3, 3)
(B1_fraction_vector <- 1 - B_fraction_vector)

#####
(num_A_fraction <- length(A_fraction_vector))
(num_A_vector <- length(A_vector))
(num_A1_vector <- length(A1_vector))
(num_A1_fraction <- length(A1_fraction_vector))
(num_B_vector <- length(B_vector))
(num_B_fraction <- length(B_fraction_vector))
(num_B1_vector <- length(B1_vector))
(num_B1_fraction <- length(B1_fraction_vector))

# num_row_table <- num_B1_vector * num_B1_fraction
(num_col_table <- num_A1_vector * num_A_fraction *  num_A_vector )
(num_row_table <- num_B1_vector * num_B_fraction *  num_B_vector)
(num_composite <- num_row_table * num_col_table)

# data.frame of different composites combined by the above components
df_elem_user = as.data.frame(matrix(nrow = num_composite, ncol = 12))
colnames(df_elem_user) = c( "A", "A_valence", "A_fraction",
                            "A1", "A1_valence", "A1_fraction",
                            "B", "B_valence", "B_fraction", 
                            "B1", "B1_valence", "B1_fraction")
df_elem_user$A <- rep(A_vector, each = num_A_fraction*num_A1_vector*num_B_fraction*num_B_vector*num_B1_vector)
df_elem_user$A_valence <- rep(A_valence_vector, each = num_A_fraction*num_A1_vector*num_B_fraction*num_B_vector*num_B1_vector)
df_elem_user$A_fraction <- rep(A_fraction_vector, each = num_B_vector*num_B_fraction*num_B1_vector, length.out = num_composite)
df_elem_user$A1 <- rep(A1_vector, each = num_A_fraction*num_B_vector*num_B_fraction*num_B1_vector, length.out = num_composite)
df_elem_user$A1_valence <- rep(A1_valence_vector, each = num_A_fraction*num_B_vector*num_B_fraction*num_B1_vector, length.out = num_composite)
df_elem_user$A1_fraction <- rep(A1_fraction_vector, each = num_B_vector*num_B_fraction*num_B1_vector, length.out = num_composite)

df_elem_user$B <- rep(B_vector, each = num_B_fraction*num_B1_vector)
df_elem_user$B_valence <- rep(B_valence_vector, each = num_B_fraction*num_B1_vector)
df_elem_user$B_fraction <- rep(B_fraction_vector, length.out = num_composite)
df_elem_user$B1 <- rep(B1_vector, each = num_B_fraction, length.out = num_composite)
df_elem_user$B1_valence <- rep(B1_valence_vector, each = num_B_fraction, length.out = num_composite)
df_elem_user$B1_fraction <- rep(B1_fraction_vector, length.out = num_composite)

df_expe_elem_user <- data.frame() 
for(i in seq(num_composite)){
  print(i)
  expe_elem_user = cbind(df_expe_user, df_elem_user[i,])
  # print(expe_elem_user)
  df_expe_elem_user = rbind(df_expe_elem_user, expe_elem_user)
} 


# Customize the descriptors based on the oxide composition and element information
source("AA1BB1_descriptor_customize.R")

## load trained XGBoost model
# load(file="saveXGBmodel_rep10cv10.rdata")
## load trained RF model
load(file="saveRFmodel_rep5cv5.rdata")


predict_pc_composites <- numeric(num_composite)
df_expe_elem_tosave <- data.frame()
start_time <-  Sys.time()
is.nan.data.frame <- function(x) do.call(cbind, lapply(x, is.nan))
for(i in seq(num_composite)){
  # i = 2
  (newdata = descriptor_customize(df_expe_elem_user[i,]))
  newdata <- within(newdata, rm("fraction_A1", "fraction_B1"))
  
  df_expe_elem_tosave = rbind(df_expe_elem_tosave, newdata)
  new_pred = predict(rf.model, newdata = newdata)
  # new_pred = predict(xgb.model, newdata = newdata)
  predict_pc_composites[i] <- max(as.numeric(new_pred$data), 0)
  
  print(i)
}
end_time <-  Sys.time()
(time_elapsed <- end_time - start_time)
# save experiment, element data and predicted PC to csv
df_expe_elem_tosave$predictedPC =  predict_pc_composites
#

predict_pc_composites <- df_expe_elem_tosave # for tuning

##
# # save predicted values with varying A-site dopants
# write.csv(predict_pc_composites, 
#           "/Users/baoyinyuan/Nutstore Files/LCN82/Script_Asite/predict_pc_composites_byDopant.csv",
#           row.names = FALSE)

# 
# predict_pc_composites %>% head(5)
# par(mfrow= c(1, 1) )
# plot(predict_pc_composites$aw_A1, predict_pc_composites$predictedPC)


# reshape the predicted PC to matrix
# predict_pc_composites_matrix <- matrix(predict_pc_composites$predictedPC, nrow = num_row_table, byrow = FALSE)
predict_pc_composites_matrix <- matrix(predict_pc_composites$predictedPC, nrow = num_B1_vector * num_B_fraction, byrow = FALSE)


# row names of matrix 
# colnames_composites_matrix <- unique(paste(df_expe_elem_user$A,
#                                            format(df_expe_elem_user$A_fraction*100, nsmall = 0),
#                                            df_expe_elem_user$A1,
#                                            format(df_expe_elem_user$A1_fraction*100, nsmall = 0),
#                                            sep = ""))

colnames_composites_matrix <- unique(paste(df_expe_elem_user$A,
                                           format(df_expe_elem_user$A_fraction*100, nsmall = 0),
                                           df_expe_elem_user$A1,
                                           format(df_expe_elem_user$A1_fraction*100, nsmall = 0),
                                           df_expe_elem_user$B,
                                           # format(df_expe_elem_user$B_fraction*100, nsmall = 0),
                                           sep = ""))

# column names of matrix
# rownames_composites_matrix <- unique(paste(df_expe_elem_user$B,
#                                            format(df_expe_elem_user$B_fraction*100, nsmall = 0),
#                                            df_expe_elem_user$B1,
#                                            format(df_expe_elem_user$B1_fraction*100, nsmall = 0),
#                                            sep = ""))

rownames_composites_matrix <- unique(paste(df_expe_elem_user$B1,
                                          format(df_expe_elem_user$B1_fraction*100, nsmall = 0),
                                          sep = ""))
rownames(predict_pc_composites_matrix) <- rownames_composites_matrix
colnames(predict_pc_composites_matrix) <- colnames_composites_matrix

## ggplot
df_predict_pc_composites <- reshape2::melt(predict_pc_composites_matrix)
df_predict_pc_composites %>% head(5)
colnames(df_predict_pc_composites) <- c("B Site","A Site","Predicted PC")

df_predict_pc_composites$`A Site` <- factor(df_predict_pc_composites$`A Site`,levels = unique(df_predict_pc_composites$`A Site`))
df_predict_pc_composites$`B Site` <- factor(df_predict_pc_composites$`B Site`,levels=unique(df_predict_pc_composites$`B Site`))

predict_pc_composites_plt <- ggplot(data = df_predict_pc_composites, aes(x = `A Site`, y = `B Site`)) +
  geom_tile(aes(fill = `Predicted PC`)) +
  xlab("Composite at A site") +
  ylab("Composite at B site") +
  scale_x_discrete(expand = c(0,0)) +
  scale_y_discrete(expand = c(0,0)) +
  scale_color_brewer(palette = "PuOr") +
  scale_fill_gradient2("Predicted\nHPC", 
                       # low = "tan", 
                       # mid = "lightgrey",
                       # high = "blue",
                       low = "tan",
                       mid = "gray95", 
                       high = "darkmagenta",
                       #midpoint = .12) + # temperature：600
                       midpoint = .09) +
  theme(axis.text.x = element_text(angle = 90, vjust = 0.5, hjust= 0.5, color="black", size = 3),
        axis.text.y = element_text(vjust = 0.5, hjust=0.5, color="black", size= 10),
        axis.title.x = element_text(face="bold", colour="black", size = 10),
        axis.title.y = element_text(face="bold", colour="black", size = 10),
        legend.key.size = unit(1, 'cm'),
        legend.text = element_text(size = 8),
        legend.title = element_text(size = 8),
        legend.position = "bottom"
  ) +
  # ggtitle(paste0("XGB", "@", df_expe_user$temperature,"oC","&",  df_expe_user$pH2O, "atm"))
  ggtitle(paste0("RF", "@", df_expe_user$temperature,"oC","&",  df_expe_user$pH2O, "atm"))

#
predict_pc_composites_plt


#
#
ggsave(filename = paste0("rf.predict_Asite_plt_AA1BB1_20240420_", df_expe_user$temperature,
                         "_", df_expe_user$pH2O,"_2rd.tiff"),
       plot = predict_pc_composites_plt,
       width = 80,  # <=19.05cm
       height = 50, # <=22.225cm
       units= "cm",
       dpi= 500,
       compression = "lzw")
# 












