# Loading images to directories
original_dataset_dir <- "C:/Users/mohbih171/Dropbox/Ehtasham/leukemia/ALL_IDB2/img"
base_dir <- "C:/Users/mohbih171/Dropbox/Ehtasham/leukemia/ALL_IDB2"


train_dir <- file.path(base_dir, "train")
dir.create(train_dir)


validation_dir <- file.path(base_dir, "validation")
dir.create(validation_dir)


test_dir <- file.path(base_dir, "test")
dir.create(test_dir)


train_no_dir <- file.path(train_dir, "no")
dir.create(train_no_dir)


train_yes_dir <- file.path(train_dir, "yes")
dir.create(train_yes_dir)


validation_no_dir <- file.path(validation_dir, "no")
dir.create(validation_no_dir)


validation_yes_dir <- file.path(validation_dir, "yes")
dir.create(validation_yes_dir)



test_no_dir <- file.path(test_dir, "no")
dir.create(test_no_dir)


test_yes_dir <- file.path(test_dir, "yes")
dir.create(test_yes_dir)



fnames <- list.files(path = original_dataset_dir, pattern = "1.tif")[1 : 100]
file.copy(file.path(original_dataset_dir, fnames), 
          file.path(train_yes_dir)) 

fnames <- list.files(path = original_dataset_dir, pattern = "1.tif")[101 : 115]
file.copy(file.path(original_dataset_dir, fnames), 
          file.path(validation_yes_dir))

fnames <- list.files(path = original_dataset_dir, pattern = "1.tif")[116 : 130]
file.copy(file.path(original_dataset_dir, fnames), 
          file.path(test_yes_dir))


fnames <- list.files(path = original_dataset_dir, pattern = "0.tif")[1 : 100]
file.copy(file.path(original_dataset_dir, fnames), 
          file.path(train_no_dir)) 

fnames <- list.files(path = original_dataset_dir, pattern = "0.tif")[101 : 115]
file.copy(file.path(original_dataset_dir, fnames), 
          file.path(validation_no_dir))

fnames <- list.files(path = original_dataset_dir, pattern = "0.tif")[116 : 130]
file.copy(file.path(original_dataset_dir, fnames), 
          file.path(test_no_dir))




# checking image distributions
cat("total training yes images:", length(list.files(train_yes_dir)), "\n")
cat("total training no images:", length(list.files(train_no_dir)), "\n")
cat("total validation yes images:", length(list.files(validation_yes_dir)), "\n")
cat("total validation no images:", length(list.files(validation_no_dir)), "\n")
cat("total test yes images:", length(list.files(test_yes_dir)), "\n")
cat("total test no images:", length(list.files(test_no_dir)), "\n")
