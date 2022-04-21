#!/bin/bash

rm submission.rpz
reprozip trace --overwrite python3 blocking_neural.py
reprozip pack submission.rpz

