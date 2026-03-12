FROM python:3.12-slim
LABEL title="hc_buckleyleverett"
LABEL description="Docker image to reproduce results in 'Efficient design of continuation methods for hyperbolic transport problems in porous media '"
LABEL version="0.1"
LABEL maintainer="Peter von Schultzendorff"
LABEL email="peter.schultzendorff@uib.no"

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    git \
    && rm -rf /var/lib/apt/lists/*

RUN git clone https://github.com/pschultzendorff/hc_buckleyleverett 

RUN pip install --no-cache-dir --upgrade pip
RUN pip install --no-cache-dir -e /app/hc_buckleyleverett

ENV MPLBACKEND=Agg

CMD ["python", "hc_buckleyleverett/scripts/buckley_leverett/viscous.py"]
