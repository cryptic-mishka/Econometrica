#!/usr/bin/env python
# coding: utf-8

# # 1 Sleep equation

# In[30]:


import numpy as np # linear algebra 
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
# stats models: regression fitting via formulas
import statsmodels.formula.api as smf
# stats models: regression fitting via matrices of regression design
import statsmodels.api as sm


# In[31]:


df = pd.read_csv('https://raw.githubusercontent.com/artamonoff/Econometrica/master/python-notebooks/data-csv/sleep75.csv')
df


# ## 1.1 Спецификация
# $$
# sleep = \beta_0+\beta_1*totwrk+\beta_2*male
# $$

# In[4]:


# специфицируем модель через формулу
cryptic_mishka = smf.ols(formula='sleep~totwrk+male', data=df).fit()
# Коэфициенты модели с округлени.
cryptic_mishka.params.round(2)


# $$
# sleep = 3573.2-0.17*totwrk+88.84*male
# $$

# 1. При увеличении количества рабочего времени на одну минуту в неделю, количество сна уменьшается на 0.17 минут в неделю, при прочих равных.
# 2. Мужчины спят на 88.84 минуты в неделю больше женщин.

# ## 1.2 Спецификация
# $$
# sleep = \beta_0+\beta_1*totwrk+\beta_2*male+\beta_3*smsa+\beta_4*age+\beta_5*south+\beta_6*yngkid+\beta_7*marr+\beta_8*union
# $$

# In[33]:


# специфицируем модель через формулу
sleep_eq2 = smf.ols(formula='sleep~totwrk+male+smsa+age+south+yngkid+marr+union', data=df).fit()
# Коэфициенты модели с округление
sleep_eq2.params.round(2)


# $$
# sleep = 3446.83-0.17*totwrk+87.11*male-54.19*smsa+2.71*age+102.27*south-13.05*yngkid+31.36*marr+11.87*union
# $$

# 1. При увеличении количества рабочего времени на одну минуту в неделю, количество сна уменьшается на 0.17 минут в неделю, при прочих равных.
# 2. Мужчины спят на 87.11 минуты в неделю больше женщин.

# ## 1.3 Спецификация
# $$
# sleep = \beta_0+\beta_1*log(hrwage)+\beta_2*smsa+\beta_3*totwrk+\beta_4*male+\beta_5*marr+\beta_6*age+\beta_7*south+\beta_8*yngkid
# $$

# In[5]:


# специфицируем модель через формулу
sleep_eq3 = smf.ols(formula='sleep~np.log(hrwage)+smsa+totwrk+male+marr+age+south+yngkid', data=df).fit()
# Коэфициенты модели с округление
sleep_eq3.params.round(2)


# $$
# sleep = 3440.19-1.39*log(hrwage)-36.96*smsa-0.16*totwrk+36.87*male+53.34*marr+2.37*age+76.27*south+47.92*yngkid
# $$

# 1. При увеличении почасовой опляты труда на 1 %, количество сна уменьшается на 0.0139 минут в неделю, при прочих равных.
# 2. Жители мегаполиса спят на 36.96 минуты в неделю меньше.

# # 2 Wage equation

# In[14]:


df = pd.read_csv('https://raw.githubusercontent.com/artamonoff/Econometrica/master/python-notebooks/data-csv/wage2.csv')
df


# ## 2.1 Спецификация
# $$
# log(wage) = \beta_0+\beta_1*age+\beta_2*IQ
# $$

# In[17]:


# специфицируем модель через формулу
wage_eq1 = smf.ols(formula='np.log(wage)~age+IQ', data=df).fit()
# Коэфициенты модели с округление
wage_eq1.params.round(3)


# $$
# log(wage) = 5.077+0.024*age+0.009*IQ
# $$

# 1. При увеличении возратса на 1 год, месячная зарплата увеличивается на 2.4 %, при прочих равных.
# 2. При увеличении IQ на 1 пункт, месячная зарплата увеличится на 0.9 %, при прочих равных.

# ## 2.2 Спецификация
# $$
# log(wage) = \beta_0+\beta_1*age+\beta_2*IQ+\beta_3*south+\beta_4*urban+\beta_5*married+\beta_6*KWW
# $$

# In[18]:


# специфицируем модель через формулу
wage_eq2 = smf.ols(formula='np.log(wage)~age+IQ+south+urban+married+KWW', data=df).fit()
# Коэфициенты модели с округление
wage_eq2.params.round(3)


# $$
# log(wage) = 5.126+0.014*age+0.007*IQ-0.101*south+0.165*urban+0.191*married+0.007*KWW
# $$

# 1. При увеличении возратса на 1 год, месячная зарплата увеличивается на 1.4 %, при прочих равных.
# 2. При увеличении IQ на 1 пункт, месячная зарплата увеличится на 0.7 %, при прочих равных.

# # 3 Output equation

# In[19]:


df = pd.read_csv('https://raw.githubusercontent.com/artamonoff/Econometrica/master/python-notebooks/data-csv/Labour.csv')
df


# ## 3.1 Спецификация
# $$
# log(output) = \beta_0+\beta_1*log(capital)+\beta_2*log(labour)
# $$

# In[20]:


# специфицируем модель через формулу
output_eq1 = smf.ols(formula='np.log(output)~np.log(capital)+np.log(labour)', data=df).fit()
# Коэфициенты модели с округление
output_eq1.params.round(3)


# $$
# log(output) = -1.711+0.208*log(capital)+0.715*log(labour)
# $$

# 1. При увеличении капитала на 1 %, выпуск увеличивается на 0.208 %, при прочих равных.
# 2. При увеличении числа сторудников на 1 %, выпуск увеличится на 0.715 %, при прочих равных.

# ## 3.2 Спецификация
# $$
# log(output) = \beta_0+\beta_1*log(capital)+\beta_2*log(labour)+\beta_3*log(wage)
# $$

# In[21]:


# специфицируем модель через формулу
output_eq2 = smf.ols(formula='np.log(output)~np.log(capital)+np.log(labour)+np.log(wage)', data=df).fit()
# Коэфициенты модели с округление
output_eq2.params.round(3)


# $$
# log(output) = -5.007+0.149*log(capital)+0.720*log(labour)+0.921*log(wage)
# $$

# 1. При увеличении капитала на 1 %, выпуск увеличивается на 0.149 %, при прочих равных.
# 2. При увеличении числа сторудников на 1 %, выпуск увеличится на 0.720 %, при прочих равных.

# # 4 Cost equation

# In[23]:


df = pd.read_csv('https://raw.githubusercontent.com/artamonoff/Econometrica/master/python-notebooks/data-csv/Electricity.csv')
df


# ## 4.1 Спецификация
# $$
# log(cost) = \beta_0+\beta_1*log(q)
# $$

# In[24]:


# специфицируем модель через формулу
cost_eq1 = smf.ols(formula='np.log(cost)~np.log(q)', data=df).fit()
# Коэфициенты модели с округление
cost_eq1.params.round(3)


# $$
# log(cost) = -3.841+0.836*log(q)
# $$

# 1. При увеличении общего выпуска электроэнергии на 1 %, общие издержки за год увеличиваются на 0.836 %, при прочих равных.

# ## 4.2 Спецификация
# $$
# log(cost) = \beta_0+\beta_1*log(q)+\beta_2*log(pl)+\beta_3*log(pf)+\beta_4*log(pk)
# $$

# In[25]:


# специфицируем модель через формулу
cost_eq2 = smf.ols(formula='np.log(cost)~np.log(q)+np.log(pl)+np.log(pf)+np.log(pk)', data=df).fit()
# Коэфициенты модели с округление
cost_eq2.params.round(3)


# $$
# log(cost) = -7.472+0.838*log(q)+0.044*log(pl)+0.713*log(pf)+0.188*log(pk)
# $$

# 1. При увеличении общего выпуска электроэнергии на 1 %, общие издержки за год увеличиваются на 0.838 %, при прочих равных.
