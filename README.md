# MovieLens Capstone

Reproducible MovieLens 10M recommender (PH125.9x). Includes report (Rmd+PDF) and a standalone R script that downloads data, builds models, and prints the final RMSE on the sealed holdout.

## Files
- `MovieLens_Capstone_Report.Rmd` – full report
- `MovieLens_Capstone_Report.pdf` – knit from the Rmd
- `movielens_capstone.R` – end-to-end script; prints FINAL RMSE and writes predictions CSV
- `knit.R` – helper to install TinyTeX and render the PDF

## Quick start
```r
# run end-to-end
source("movielens_capstone.R", echo = TRUE)

# knit report to PDF
source("knit.R")
# or: rmarkdown::render("MovieLens_Capstone_Report.Rmd", "pdf_document")
