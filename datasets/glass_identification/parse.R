require(data.table)
require(mldata.export)

main_dir <- "./data_raw/glass_identification"
write_dir <- main_dir

dataset_name <- "glass_identification"

#### Data preparation ----------------------------------------------------------

data <- data.table::fread(input=file.path(main_dir, "data", "glass.data.csv"))

# Set names
data.table::setnames(
  data,
  old=colnames(data),
  new=c(
    "id_number", "refractive_index", "sodium_oxide_weight_percent",
    "magnesium_oxide_weight_percent", "aluminium_oxide_weight_percent", "silicon_oxide_weight_percent",
    "potassium_oxide_weight_percent", "calcium_oxide_weight_percent", "barium_oxide_weight_percent",
    "iron_oxide_weight_percent", "target")
)

# Drop id_number
data[, "id_number":=NULL]

# Encode categorical variables.
data$target <- factor(x=data$target, levels=c(1, 2, 3, 5, 6, 7), labels=c(
  "window float glass", "window non-float glass", "vehicle window float glass",
  "container", "tableware", "headlamp"
))

default_descriptors <- list(
  refractive_index = "Refractive index.",
  sodium_oxide_weight_percent = "Sodium oxide (Na2O) content of glass sample in weight percent.",
  magnesium_oxide_weight_percent = "Magnesium oxide (MgO) content of glass sample in weight percent.",
  aluminium_oxide_weight_percent = "Aluminium oxide (Al2O3) content of glass sample in weight percent.",
  silicon_oxide_weight_percent =  "Silicon dioxide (SiO2) content of glass sample in weight percent. This is generally the main component of glass.",
  potassium_oxide_weight_percent = "Potassium oxide (K2O) content of glass sample in weight percent.",
  calcium_oxide_weight_percent = "Calcium oxide (CaO) content of glass sample in weight percent.",
  barium_oxide_weight_percent = "Barium oxide (BaO) content of glass sample in weight percent.",
  iron_oxide_weight_percent = "Iron oxide (Fe2O3) content of glass sample in weight percent.",
  target = "Origin of glass sample."
)

#### Metadata ------------------------------------------------------------------
# Create and write metadata
metadata <- mldata.export::create_metadata(
  name = dataset_name,
  data = data,
  descriptors = default_descriptors
)

# Data is from the UCI ML repository.
metadata <- mldata.export::use_source_uci_ml_repository(metadata)

# Set title
metadata$title <- "glass origin prediction"

metadata$description <- paste0(
  "Glass left at a crime scene may provide evidence, if the origins of the class can be determined. ",
  "In this dataset, glass samples were analysed according to material properties, particulary ",
  "their oxide contents. The task is to predict the origin of the glass samples according to their properties."
)

metadata$source <- c(
  "https://archive-beta.ics.uci.edu/ml/datasets/contraceptive+method+choice",
  "B. German, Central Research Establishment, Home Office Forensic Science Service, Aldermaston, Reading, Berkshire RG7 4PN",
  "Vina Spiehler, Ph.D., DABFT Diagnostic Products Corporation (213) 776-0180 (ext 3014)"
)

metadata$keywords <- c("material", "forensics")

mldata.export::write_metadata(
  metadata,
  path=file.path(write_dir, dataset_name)
)

#### Summary statistics --------------------------------------------------------
mldata.export::write_summary_stats(
  data=data,
  metadata=metadata,
  path=file.path(write_dir, dataset_name)
)

#### Data ----------------------------------------------------------------------
mldata.export::write_data(
  data=data,
  name=dataset_name,
  path=file.path(write_dir, dataset_name)
)
