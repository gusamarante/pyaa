# pyAA
All around finance resources. This library started as an asset 
allocation project (hence the name, pyAA) but evolved to include 
more subjects and lecture notes.

---

# Main Features
- **Portfolio Construction**
  - [Mean-Variance Optimization](https://github.com/gusamarante/pyaa/blob/0cd7e4896856fe51a6832bd41edfa2adc9be9eda/allocation.py#L15)
  - [Hierarchical Risk Parity](https://github.com/gusamarante/pyaa/blob/f86f482913d08d525fac889f61d30afe6c079d48/allocation.py#L341)
  - [Risk Budget / Equal Risk Contribution](https://github.com/gusamarante/pyaa/blob/f86f482913d08d525fac889f61d30afe6c079d48/allocation.py#L544)
  - [Black-Litterman](https://github.com/gusamarante/pyaa/blob/0cd7e4896856fe51a6832bd41edfa2adc9be9eda/allocation.py#L610)
- **Pricing**
  - [Yield Curve Bootstrapp](https://github.com/gusamarante/pyaa/blob/0cd7e4896856fe51a6832bd41edfa2adc9be9eda/curves/bootstrap.py#L7)
- **Models**
  - [ACM Term Premium](https://github.com/gusamarante/pyaa/blob/0cd7e4896856fe51a6832bd41edfa2adc9be9eda/models/acm.py#L8)
- **Data Sources**
  - Federal Reserve Economic Data: [code](https://github.com/gusamarante/pyaa/blob/0cd7e4896856fe51a6832bd41edfa2adc9be9eda/data/data_api.py#L97) and [source](https://fred.stlouisfed.org/)
  - Brazilian Central Bank: [code](https://github.com/gusamarante/pyaa/blob/0cd7e4896856fe51a6832bd41edfa2adc9be9eda/data/data_api.py#L4) and [source](https://www3.bcb.gov.br/sgspub/localizarseries/localizarSeries.do?method=prepararTelaLocalizarSeries)
---

# Data Sources
Most of the data used in this repository comes from proprietary 
sources. They only work for my specific python environment. 
Although the data is not available for you, all the logic of 
the code is.
