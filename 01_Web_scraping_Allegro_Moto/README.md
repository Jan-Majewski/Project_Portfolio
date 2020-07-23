
<!-- PROJECT LOGO -->


<br />
<p align="center">
  <a href="https://github.com/Jan-Majewski/Project_Portfolio/01_Web_scraping_Allegro_Moto">
    <img src="images/logo.png" alt="Logo" width="642" height="481">
  </a>



</p>

-->

# BMW price analysis with Allegro web-scraping




<!-- Add buttons here -->

![GitHub last commit](https://img.shields.io/github/last-commit/Jan-Majewski/Project_Portfolio?01_Web_scraping_Allegro_Moto)
[![LinkedIn][linkedin-shield]][linkedin-url]






<!-- TABLE OF CONTENTS -->
## Table of Contents

* [About the Project](#about-the-project)
* [Built With](#built-with)
<<<<<<< HEAD
* [Key Takeaways](#key-takeaways)
=======
>>>>>>> 2894dcd43ffe7ab6dc8e1b4aa4feb4ec5d2ea1e1





<!-- ABOUT THE PROJECT -->
## About The Project

The aim of this project was to practice Web Scraping, EDA and simple regression models using real data. I chose car offers from Allegro (leading online marketplace in Poland), focusing on BMW out of personal interest.

#### The project consists of 2 parts:

* PART 1) Webscraping - notebook focused on scraping Allegro pages with beautiful soup, understanding the html code and extracting desired data. The aim of this notebook is to serve as a tutorial to data extraction from Allegro and can be easility used for any Automotive Allegro pages.

* PART 2) EDA and price benchmarking - exploring 2000 BMW car offers scrapped from Allegro. Initial feature exploration, analysing key depraciation drivers and transforming data is followed by building a simple regression model, which allows to interpret key features effect on offer price. Simple linear regression model on log price provides quite an accurate estimate of key features such as Age, Mileage and Engine Power on offer price. 



### Built With

* BeautifulSoup
* SciKit-learn
* Plotly
* Statsmodels


### Key takeaways
A simple econometric approach to price benchmarking with use of WLS regression provided interesting insights into 8 key features with greatest influence on car price and depreciation. Although the model itself was underfitting the data and had mean percentage error of over 35% on train set, the coefficients provided some interesting trends, which moved in line with key drivers based on domain knowledge. 

As an example, coefficient for Used feature was equal to -0.22, which can be interpreted as a BMW car loosing 22% of its value after leaving the dealership - knowing that this fall also includes dealership discounts the value is quite precise. 

Age and Mileage have similar efect on price with approximately 16% of value lost for every 100k km driven and 8.5% depreciation with every year of age. 

There were only 3 features increasing offer price - 20% increase for 4-wheel drive, 28% of increase for automatic gearbox and 32% increase for every extra 100 bhp. 

High price increase for automatic gearbox is probably the most missleading coefficient, but knowing that this type of gearbox is usually present in higher trims it could be justified as premium paid for the highest configuration, which can reach over 20% in premium cars. 



[linkedin-shield]: https://img.shields.io/badge/-LinkedIn-black.svg?style=flat-square&logo=linkedin&colorB=555
[linkedin-url]: https://www.linkedin.com/in/jan-majewski-132907104/
