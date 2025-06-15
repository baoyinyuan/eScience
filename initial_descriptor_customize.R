#######
# Based on the oxide-composition and the element information to customize the descriptors,
# which would be used for the following model training
#######
descriptor_customize <- function(AA1BB1O3_sample){
  
  # AA1BB1O3_sample = df_expe_elem
  # AA1BB1O3_sample = df_expe_elem_user[i,]
  # AA1BB1O3_sample = df_expe
  
  # nbr of samples
  (N = dim(AA1BB1O3_sample)[1]) 
  
  # fraction of composition
  (fraction_A = AA1BB1O3_sample['A_fraction'])
  (fraction_A1 = AA1BB1O3_sample['A1_fraction'])
  (fraction_B = AA1BB1O3_sample['B_fraction'])
  (fraction_B1 = AA1BB1O3_sample['B1_fraction'])
  
  # extract element attribute of A-site host element in composition
  aw_A = numeric()
  ad_A = numeric()
  mp_A = numeric()
  fie_A = numeric()
  eln_A = numeric()
  ir_A = numeric()
  for (n in seq(N)){
    if (fraction_A[n,] > 0){
      a = AA1BB1O3_sample['A'][n,]
      v = AA1BB1O3_sample['A_valence'][n,]
      i = which((df_elem['Atom'] == a) & (df_elem['Valence'] == v))
      if(length(i) == 0){
        print("Invalid element!")
      }
      aw_A[n] = df_elem['Atomic.weight'][i,]
      ad_A[n] = df_elem['Atomic.density'][i,]
      mp_A[n] = df_elem['Melting.point'][i,] 
      fie_A[n] = df_elem['First.ionization.energy'][i,]
      eln_A[n] = df_elem['Electronegativity'][i,]
      ir_A[n] = df_elem['Ionic.radius.VI'][i,]
    }
  }
  
  
  # extract element attribute of A-site dopant element in composition
  aw_A1 = numeric()
  ad_A1 = numeric()
  mp_A1 = numeric()
  fie_A1 = numeric()
  eln_A1 = numeric()
  ir_A1 = numeric()
  for (n in seq(N)){
    if ((fraction_A1[n,] == 0) | (is.na(fraction_A1[n,]))){
      aw_A1[n] = 0
      ad_A1[n] = 0
      mp_A1[n] = 0
      fie_A1[n] = 0
      eln_A1[n] = 0
      ir_A1[n] = 0
    } else if (fraction_A1[n,] > 0){
      a = AA1BB1O3_sample["A1"][n,]
      v = AA1BB1O3_sample['A1_valence'][n,]
      i = which((df_elem['Atom'] == a) & (df_elem['Valence'] == v))
      if(length(i) == 0){
        print("Invalid element!")
      }
      aw_A1[n] = df_elem['Atomic.weight'][i,]
      ad_A1[n] = df_elem['Atomic.density'][i,]
      mp_A1[n] = df_elem['Melting.point'][i,] 
      fie_A1[n] = df_elem['First.ionization.energy'][i,]
      eln_A1[n] = df_elem['Electronegativity'][i,]
      ir_A1[n] = df_elem['Ionic.radius.VI'][i,]
    }
  }
  
  
  # extract element attribute of B-site host element in composition
  aw_B = numeric()
  ad_B = numeric()
  mp_B = numeric()
  fie_B = numeric()
  eln_B = numeric()
  ir_B = numeric()
  for (n in seq(N)){
    if ((fraction_B[n,] == 0) | (is.na(fraction_B[n,]))){
      next
    } else if ((fraction_B[n,] > 0)){
      a = AA1BB1O3_sample['B'][n,]
      v = AA1BB1O3_sample['B_valence'][n,]
      i = which((df_elem['Atom'] == a) & (df_elem['Valence'] == v))
      if(length(i) == 0){
        print("Invalid element!")
      }
      aw_B[n] = df_elem['Atomic.weight'][i,]
      ad_B[n] = df_elem['Atomic.density'][i,]
      mp_B[n] = df_elem['Melting.point'][i,] 
      fie_B[n] = df_elem['First.ionization.energy'][i,]
      eln_B[n] = df_elem['Electronegativity'][i,]
      ir_B[n] = df_elem['Ionic.radius.VI'][i,]
    }
  }
  
  # extract element attribute of B-site dopant element in composition
  aw_B1 = numeric()
  ad_B1 = numeric()
  mp_B1 = numeric()
  fie_B1 = numeric()
  eln_B1 = numeric()
  ir_B1 = numeric()
  for (n in seq(N)){
    if ((fraction_B1[n,] == 0) | (is.na(fraction_B1[n,]))){
      aw_B1[n] = 0
      ad_B1[n] = 0
      mp_B1[n] = 0
      fie_B1[n] = 0
      eln_B1[n] = 0
      ir_B1[n] = 0
      } else if ((fraction_B1[n,] > 0)){
      a = AA1BB1O3_sample['B1'][n,]
      v = AA1BB1O3_sample['B1_valence'][n,]
      i = which((df_elem['Atom'] == a) & (df_elem['Valence'] == v))
      if(length(i) == 0){
        print(n)
        print("Invalid element!")
      }
      aw_B1[n] = df_elem['Atomic.weight'][i,]
      ad_B1[n] = df_elem['Atomic.density'][i,]
      mp_B1[n] = df_elem['Melting.point'][i,] 
      fie_B1[n] = df_elem['First.ionization.energy'][i,]
      eln_B1[n] = df_elem['Electronegativity'][i,]
      ir_B1[n] = df_elem['Ionic.radius.VI'][i,]
    }
  }
  
  ######
  # ## artificial descriptors
  # formular_weight_oxides <- fraction_A * aw_A + fraction_A1 * aw_A1 + fraction_B * aw_B +
  #   fraction_B1 * aw_B1 + 48
  # ave_cation_valence <- fraction_A * AA1BB1O3_sample$A_valence + fraction_A1 * AA1BB1O3_sample$A1_valence +
  #   fraction_B * AA1BB1O3_sample$B_valence + fraction_B1 * AA1BB1O3_sample$B1_valence
  # ave_cation_valence_A_site <- fraction_A * AA1BB1O3_sample$A_valence + fraction_A1 * AA1BB1O3_sample$A1_valence
  # ave_cation_valence_B_site <- fraction_B * AA1BB1O3_sample$B_valence + fraction_B1 * AA1BB1O3_sample$B1_valence
  # ave_cation_eln <- fraction_A * eln_A + fraction_A1 * eln_A1 + fraction_B * eln_B + fraction_B1 * eln_B1
  # ave_cation_eln_A_site <- fraction_A * eln_A + fraction_A1 * eln_A1
  # ave_cation_eln_B_site <- fraction_B * eln_B + fraction_B1 * eln_B1
  # tolerance_factor <- (fraction_A * ir_A + fraction_A1 * ir_A1 + 1.4)/(sqrt(2)*(fraction_B * ir_B + fraction_B1 * ir_B1 + 1.4))
  # ave_cation_ir <- fraction_A * ir_A + fraction_A1 * ir_A1 + fraction_B * ir_B + fraction_B1 * ir_B1
  # ave_cation_ir_A_site <- fraction_A * ir_A + fraction_A1 * ir_A1
  # ave_cation_ir_B_site <- fraction_B * ir_B + fraction_B1 * ir_B1
  # ave_cation_mp <- fraction_A * mp_A + fraction_A1 * mp_A1 + fraction_B * mp_B + fraction_B1 * mp_B1
  # ave_cation_mp_A_site <- fraction_A * mp_A + fraction_A1 * mp_A1
  # ave_cation_mp_B_site <- fraction_B * mp_B + fraction_B1 * mp_B1
  # ave_cation_aw <- fraction_A * aw_A + fraction_A1 * aw_A1 + fraction_B * aw_B + fraction_B1 * aw_B1
  # ave_cation_aw_A_site <- fraction_A * aw_A + fraction_A1 * aw_A1
  # ave_cation_aw_B_site <- fraction_B * aw_B + fraction_B1 * aw_B1
  # ave_cation_fie <- fraction_A * fie_A + fraction_A1 * fie_A1 + fraction_B * fie_B + fraction_B1 * fie_B1
  # ave_cation_fie_A_site <- fraction_A * fie_A + fraction_A1 * fie_A1
  # ave_cation_fie_B_site <- fraction_B * fie_B + fraction_B1 * fie_B1
  # sum_fraction_A1B1 <- fraction_A1 + fraction_B1
  # ratio_fraction_A1B1_AB <- sum_fraction_A1B1/2
  # ratio_fraction_A1_A <- fraction_A1/fraction_A
  # ratio_fraction_B1_B <- fraction_B1/fraction_B
  # ratio_ir_A1_A <- ir_A1/ir_A
  # ratio_ir_B1_B <- ir_B1/ir_B
  # ratio_mp_A1_A <- mp_A1/mp_A
  # ratio_mp_B1_B <- mp_B1/mp_B
  # ratio_fie_A1_A <- fie_A1/fie_A
  # ratio_fie_B1_B <- fie_B1/fie_B
  # ratio_eln_A1_A <- eln_A1/eln_A
  # ratio_eln_B1_B <- eln_B1/eln_B
  # ratio_ir_A_B <- ir_A/ir_B
  # ratio_mp_A_B <- mp_A/mp_B
  # ratio_fie_A_B <- fie_A/fie_B
  # ratio_eln_A_B <- eln_A/eln_B
  
  ## merge all the new descriptors into one dataframe
  # df_descriptor <- data.frame(formular_weight_oxides, ave_cation_valence, ave_cation_valence_A_site, 
  #                             ave_cation_valence_B_site, ave_cation_eln, ave_cation_eln_A_site,
  #                             ave_cation_eln_B_site, tolerance_factor, ave_cation_ir, ave_cation_ir_A_site,
  #                             ave_cation_ir_B_site, ave_cation_mp, ave_cation_mp_A_site, ave_cation_mp_B_site,
  #                             ave_cation_aw, ave_cation_aw_A_site, ave_cation_aw_B_site, ave_cation_fie,
  #                             ave_cation_fie_A_site, ave_cation_fie_B_site, sum_fraction_A1B1,ratio_fraction_A1B1_AB,
  #                             ratio_fraction_A1_A, ratio_fraction_B1_B, ratio_ir_A1_A, ratio_ir_B1_B, ratio_mp_A1_A,
  #                             ratio_mp_B1_B, ratio_fie_A1_A, ratio_fie_B1_B, ratio_eln_A1_A, ratio_eln_B1_B,
  #                             ratio_ir_A_B, ratio_mp_A_B, ratio_fie_A_B, ratio_eln_A_B)
  # colnames(df_descriptor) <- c("formular_weight_oxides", "ave_cation_valence", "ave_cation_valence_A_site", 
  #                              "ave_cation_valence_B_site", "ave_cation_eln", "ave_cation_eln_A_site",
  #                              "ave_cation_eln_B_site", "tolerance_factor", "ave_cation_ir", "ave_cation_ir_A_site",
  #                              "ave_cation_ir_B_site", "ave_cation_mp", "ave_cation_mp_A_site", "ave_cation_mp_B_site",
  #                              "ave_cation_aw", "ave_cation_aw_A_site", "ave_cation_aw_B_site", "ave_cation_fie",
  #                              "ave_cation_fie_A_site", "ave_cation_fie_B_site", "sum_fraction_A1B1", "ratio_fraction_A1B1_AB",
  #                              "ratio_fraction_A1_A", "ratio_fraction_B1_B", "ratio_ir_A1_A", "ratio_ir_B1_B", "ratio_mp_A1_A",
  #                              "ratio_mp_B1_B", "ratio_fie_A1_A", "ratio_fie_B1_B", "ratio_eln_A1_A", "ratio_eln_B1_B",
  #                              "ratio_ir_A_B", "ratio_mp_A_B","ratio_fie_A_B", "ratio_eln_A_B")
########
  
  
  
  # element attributes of A site,A1 site, B site and B1 site
  df_elem_attr <- data.frame(aw_A, ad_A, mp_A, fie_A, eln_A,ir_A,
                             aw_A1, ad_A1, mp_A1, fie_A1, eln_A1,ir_A1,
                             aw_B, ad_B, mp_B, fie_B, eln_B,ir_B,
                             aw_B1, ad_B1, mp_B1, fie_B1, eln_B1,ir_B1)
  
  colnames(df_elem_attr) <- c("aw_A", "ad_A", "mp_A", "fie_A", "eln_A", "ir_A",
                              "aw_A1", "ad_A1", "mp_A1", "fie_A1", "eln_A1", "ir_A1",
                              "aw_B", "ad_B", "mp_B", "fie_B", "eln_B", "ir_B",
                              "aw_B1", "ad_B1", "mp_B1", "fie_B1", "eln_B1", "ir_B1")
  
  
  # experimental conditions
  feature_names = c("temperature", "pH2O", "A_fraction",
                    "A1_fraction", "B_fraction", "B1_fraction")
  
  
  df_expe_slected = AA1BB1O3_sample[feature_names]
  colnames(df_expe_slected) = c("temperature", "pH2O", "fraction_A",
                                "fraction_A1", "fraction_B", "fraction_B1")
  
  #
  # combine the experimental conditions, the element attributes and calculated element descriptors
  # df_sample = cbind(df_expe_slected, df_elem_attr, df_descriptor)
  df_sample = cbind(df_expe_slected, df_elem_attr)
  # 
  
  return(df_sample)  
}

