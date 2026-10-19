# FP16, DiffKV, DiffKV-v1 성능 비교

각 B, ISL, OSL 조건에서 FP16, DiffKV, DiffKV-v1 순으로 표시했습니다. 시간 단위는 초, 소수점 5자리이며, 괄호는 전체 생성 시간 대비 비중입니다.

FP16과 DiffKV-v1은 profiling-TOBE-table.md의 3회 측정 평균값과 평균 비중을 사용하고 편차 표시는 생략했습니다. DiffKV는 profiling-DIffKV-table.md의 최적화 전 단일 측정값을 그대로 사용했습니다.

<table style="border-collapse: collapse;">
  <thead>
    <tr>
      <th align="right">B</th>
      <th align="right">ISL</th>
      <th align="right">OSL</th>
      <th align="left">Config</th>
      <th align="right">E2E latency (s)</th>
      <th align="right">GEMM</th>
      <th align="right">Attention</th>
      <th align="right">KV Update</th>
      <th align="right">Other GPU kernels</th>
      <th align="right">Others</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td rowspan="3" align="right" valign="middle">8</td>
      <td rowspan="3" align="right" valign="middle">512</td>
      <td rowspan="3" align="right" valign="middle">512</td>
      <td align="left">FP16</td>
      <td align="right"><strong>14.75222</strong></td>
      <td align="right">11.99310 (81.30%)</td>
      <td align="right"><strong>0.81314 (5.51%)</strong></td>
      <td align="right"><strong>0.06892 (0.47%)</strong></td>
      <td align="right"><strong>0.52012 (3.53%)</strong></td>
      <td align="right"><strong>1.34159 (9.09%)</strong></td>
    </tr>
    <tr>
      <td align="left">DiffKV</td>
      <td align="right">20.97808</td>
      <td align="right">12.01111 (57.26%)</td>
      <td align="right">1.61133 (7.68%)</td>
      <td align="right">0.53703 (2.56%)</td>
      <td align="right">0.56564 (2.70%)</td>
      <td align="right">6.21600 (29.63%)</td>
    </tr>
    <tr>
      <td align="left">DiffKV-v1</td>
      <td align="right">16.19995</td>
      <td align="right">12.00980 (74.13%)</td>
      <td align="right">1.61037 (9.94%)</td>
      <td align="right">0.53715 (3.32%)</td>
      <td align="right">0.56690 (3.50%)</td>
      <td align="right">1.45912 (9.01%)</td>
    </tr>
    <tr>
      <td rowspan="3" align="right" valign="middle" style="border-top: 2px solid #666;">8</td>
      <td rowspan="3" align="right" valign="middle" style="border-top: 2px solid #666;">512</td>
      <td rowspan="3" align="right" valign="middle" style="border-top: 2px solid #666;">1024</td>
      <td align="left" style="border-top: 2px solid #666;">FP16</td>
      <td align="right" style="border-top: 2px solid #666;"><strong>29.37231</strong></td>
      <td align="right" style="border-top: 2px solid #666;">22.97562 (78.22%)</td>
      <td align="right" style="border-top: 2px solid #666;"><strong>2.16568 (7.37%)</strong></td>
      <td align="right" style="border-top: 2px solid #666;"><strong>0.13382 (0.46%)</strong></td>
      <td align="right" style="border-top: 2px solid #666;"><strong>1.00703 (3.43%)</strong></td>
      <td align="right" style="border-top: 2px solid #666;">3.05761 (10.41%)</td>
    </tr>
    <tr>
      <td align="left">DiffKV</td>
      <td align="right">44.28880</td>
      <td align="right">23.01161 (51.96%)</td>
      <td align="right">3.92545 (8.86%)</td>
      <td align="right">1.10340 (2.49%)</td>
      <td align="right">1.09082 (2.46%)</td>
      <td align="right">15.08363 (34.06%)</td>
    </tr>
    <tr>
      <td align="left">DiffKV-v1</td>
      <td align="right">32.07058</td>
      <td align="right">23.00835 (71.74%)</td>
      <td align="right">3.92067 (12.23%)</td>
      <td align="right">1.10359 (3.44%)</td>
      <td align="right">1.09630 (3.42%)</td>
      <td align="right"><strong>2.90728 (9.07%)</strong></td>
    </tr>
    <tr>
      <td rowspan="3" align="right" valign="middle" style="border-top: 2px solid #666;">8</td>
      <td rowspan="3" align="right" valign="middle" style="border-top: 2px solid #666;">512</td>
      <td rowspan="3" align="right" valign="middle" style="border-top: 2px solid #666;">2048</td>
      <td align="left" style="border-top: 2px solid #666;">FP16</td>
      <td align="right" style="border-top: 2px solid #666;"><strong>60.71259</strong></td>
      <td align="right" style="border-top: 2px solid #666;">44.94159 (74.02%)</td>
      <td align="right" style="border-top: 2px solid #666;"><strong>6.47459 (10.66%)</strong></td>
      <td align="right" style="border-top: 2px solid #666;"><strong>0.26359 (0.43%)</strong></td>
      <td align="right" style="border-top: 2px solid #666;"><strong>1.98167 (3.26%)</strong></td>
      <td align="right" style="border-top: 2px solid #666;">6.97720 (11.49%)</td>
    </tr>
    <tr>
      <td align="left">DiffKV</td>
      <td align="right">85.70383</td>
      <td align="right">45.02137 (52.53%)</td>
      <td align="right">10.55440 (12.31%)</td>
      <td align="right">2.34731 (2.74%)</td>
      <td align="right">2.14947 (2.51%)</td>
      <td align="right">25.48171 (29.73%)</td>
    </tr>
    <tr>
      <td align="left">DiffKV-v1</td>
      <td align="right">65.30314</td>
      <td align="right">45.01738 (68.94%)</td>
      <td align="right">10.54312 (16.14%)</td>
      <td align="right">2.34794 (3.60%)</td>
      <td align="right">2.15783 (3.30%)</td>
      <td align="right"><strong>5.16310 (7.91%)</strong></td>
    </tr>
    <tr>
      <td rowspan="3" align="right" valign="middle" style="border-top: 2px solid #666;">8</td>
      <td rowspan="3" align="right" valign="middle" style="border-top: 2px solid #666;">1024</td>
      <td rowspan="3" align="right" valign="middle" style="border-top: 2px solid #666;">512</td>
      <td align="left" style="border-top: 2px solid #666;">FP16</td>
      <td align="right" style="border-top: 2px solid #666;"><strong>16.68797</strong></td>
      <td align="right" style="border-top: 2px solid #666;">12.96708 (77.70%)</td>
      <td align="right" style="border-top: 2px solid #666;"><strong>1.39015 (8.33%)</strong></td>
      <td align="right" style="border-top: 2px solid #666;"><strong>0.07293 (0.44%)</strong></td>
      <td align="right" style="border-top: 2px solid #666;"><strong>0.55375 (3.32%)</strong></td>
      <td align="right" style="border-top: 2px solid #666;">1.68684 (10.11%)</td>
    </tr>
    <tr>
      <td align="left">DiffKV</td>
      <td align="right">22.71153</td>
      <td align="right">12.98494 (57.17%)</td>
      <td align="right">2.32071 (10.22%)</td>
      <td align="right">0.57797 (2.54%)</td>
      <td align="right">0.60047 (2.64%)</td>
      <td align="right">6.18954 (27.25%)</td>
    </tr>
    <tr>
      <td align="left">DiffKV-v1</td>
      <td align="right">17.85045</td>
      <td align="right">12.98329 (72.73%)</td>
      <td align="right">2.31687 (12.98%)</td>
      <td align="right">0.57813 (3.24%)</td>
      <td align="right">0.60217 (3.37%)</td>
      <td align="right"><strong>1.35215 (7.57%)</strong></td>
    </tr>
    <tr>
      <td rowspan="3" align="right" valign="middle" style="border-top: 2px solid #666;">8</td>
      <td rowspan="3" align="right" valign="middle" style="border-top: 2px solid #666;">1024</td>
      <td rowspan="3" align="right" valign="middle" style="border-top: 2px solid #666;">1024</td>
      <td align="left" style="border-top: 2px solid #666;">FP16</td>
      <td align="right" style="border-top: 2px solid #666;"><strong>32.03516</strong></td>
      <td align="right" style="border-top: 2px solid #666;">23.94978 (74.76%)</td>
      <td align="right" style="border-top: 2px solid #666;"><strong>3.26205 (10.18%)</strong></td>
      <td align="right" style="border-top: 2px solid #666;"><strong>0.13786 (0.43%)</strong></td>
      <td align="right" style="border-top: 2px solid #666;"><strong>1.04085 (3.25%)</strong></td>
      <td align="right" style="border-top: 2px solid #666;">3.60786 (11.26%)</td>
    </tr>
    <tr>
      <td align="left">DiffKV</td>
      <td align="right">44.02401</td>
      <td align="right">23.98836 (54.49%)</td>
      <td align="right">5.28530 (12.01%)</td>
      <td align="right">1.18013 (2.68%)</td>
      <td align="right">1.12913 (2.56%)</td>
      <td align="right">12.36620 (28.09%)</td>
    </tr>
    <tr>
      <td align="left">DiffKV-v1</td>
      <td align="right">34.18960</td>
      <td align="right">23.98634 (70.16%)</td>
      <td align="right">5.27778 (15.44%)</td>
      <td align="right">1.18069 (3.45%)</td>
      <td align="right">1.13225 (3.31%)</td>
      <td align="right"><strong>2.57634 (7.54%)</strong></td>
    </tr>
    <tr>
      <td rowspan="3" align="right" valign="middle" style="border-top: 2px solid #666;">8</td>
      <td rowspan="3" align="right" valign="middle" style="border-top: 2px solid #666;">1024</td>
      <td rowspan="3" align="right" valign="middle" style="border-top: 2px solid #666;">2048</td>
      <td align="left" style="border-top: 2px solid #666;">FP16</td>
      <td align="right" style="border-top: 2px solid #666;"><strong>64.89921</strong></td>
      <td align="right" style="border-top: 2px solid #666;">45.91589 (70.75%)</td>
      <td align="right" style="border-top: 2px solid #666;"><strong>8.71474 (13.43%)</strong></td>
      <td align="right" style="border-top: 2px solid #666;"><strong>0.26763 (0.41%)</strong></td>
      <td align="right" style="border-top: 2px solid #666;"><strong>2.01539 (3.11%)</strong></td>
      <td align="right" style="border-top: 2px solid #666;">7.90277 (12.18%)</td>
    </tr>
    <tr>
      <td align="left">DiffKV</td>
      <td align="right">89.82067</td>
      <td align="right">46.00682 (51.22%)</td>
      <td align="right">13.18390 (14.68%)</td>
      <td align="right">2.49449 (2.78%)</td>
      <td align="right">2.18565 (2.43%)</td>
      <td align="right">25.79745 (28.72%)</td>
    </tr>
    <tr>
      <td align="left">DiffKV-v1</td>
      <td align="right">69.53731</td>
      <td align="right">46.00176 (66.16%)</td>
      <td align="right">13.16345 (18.93%)</td>
      <td align="right">2.49482 (3.59%)</td>
      <td align="right">2.19269 (3.15%)</td>
      <td align="right"><strong>5.60485 (8.06%)</strong></td>
    </tr>
    <tr>
      <td rowspan="3" align="right" valign="middle" style="border-top: 2px solid #666;">8</td>
      <td rowspan="3" align="right" valign="middle" style="border-top: 2px solid #666;">2048</td>
      <td rowspan="3" align="right" valign="middle" style="border-top: 2px solid #666;">512</td>
      <td align="left" style="border-top: 2px solid #666;">FP16</td>
      <td align="right" style="border-top: 2px solid #666;"><strong>20.49245</strong></td>
      <td align="right" style="border-top: 2px solid #666;">14.97229 (73.06%)</td>
      <td align="right" style="border-top: 2px solid #666;"><strong>2.61583 (12.76%)</strong></td>
      <td align="right" style="border-top: 2px solid #666;"><strong>0.08111 (0.40%)</strong></td>
      <td align="right" style="border-top: 2px solid #666;"><strong>0.62096 (3.03%)</strong></td>
      <td align="right" style="border-top: 2px solid #666;">2.18018 (10.64%)</td>
    </tr>
    <tr>
      <td align="left">DiffKV</td>
      <td align="right">26.85654</td>
      <td align="right">14.99662 (55.84%)</td>
      <td align="right">3.80688 (14.17%)</td>
      <td align="right">0.67492 (2.51%)</td>
      <td align="right">0.67352 (2.51%)</td>
      <td align="right">6.66665 (24.82%)</td>
    </tr>
    <tr>
      <td align="left">DiffKV-v1</td>
      <td align="right">21.57474</td>
      <td align="right">14.99555 (69.51%)</td>
      <td align="right">3.80404 (17.63%)</td>
      <td align="right">0.67498 (3.13%)</td>
      <td align="right">0.67541 (3.13%)</td>
      <td align="right"><strong>1.40404 (6.51%)</strong></td>
    </tr>
  </tbody>
</table>

<table style="border-collapse: collapse;">
  <thead>
    <tr>
      <th align="right">B</th>
      <th align="right">ISL</th>
      <th align="right">OSL</th>
      <th align="left">Config</th>
      <th align="right">E2E latency (s)</th>
      <th align="right">GEMM</th>
      <th align="right">Attention</th>
      <th align="right">KV Update</th>
      <th align="right">Other GPU kernels</th>
      <th align="right">Others</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td rowspan="3" align="right" valign="middle">8</td>
      <td rowspan="3" align="right" valign="middle">2048</td>
      <td rowspan="3" align="right" valign="middle">1024</td>
      <td align="left">FP16</td>
      <td align="right"><strong>37.32511</strong></td>
      <td align="right">25.95461 (69.54%)</td>
      <td align="right"><strong>5.60514 (15.02%)</strong></td>
      <td align="right"><strong>0.14602 (0.39%)</strong></td>
      <td align="right"><strong>1.10844 (2.97%)</strong></td>
      <td align="right">4.46474 (11.96%)</td>
    </tr>
    <tr>
      <td align="left">DiffKV</td>
      <td align="right">51.16633</td>
      <td align="right">26.00730 (50.83%)</td>
      <td align="right">8.03733 (15.71%)</td>
      <td align="right">1.34292 (2.62%)</td>
      <td align="right">1.20093 (2.35%)</td>
      <td align="right">14.50076 (28.34%)</td>
    </tr>
    <tr>
      <td align="left">DiffKV-v1</td>
      <td align="right">39.37202</td>
      <td align="right">26.00548 (66.05%)</td>
      <td align="right">8.03182 (20.40%)</td>
      <td align="right">1.34326 (3.41%)</td>
      <td align="right">1.20585 (3.06%)</td>
      <td align="right"><strong>2.74290 (6.97%)</strong></td>
    </tr>
    <tr>
      <td rowspan="3" align="right" valign="middle" style="border-top: 2px solid #666;">8</td>
      <td rowspan="3" align="right" valign="middle" style="border-top: 2px solid #666;">2048</td>
      <td rowspan="3" align="right" valign="middle" style="border-top: 2px solid #666;">2048</td>
      <td align="left" style="border-top: 2px solid #666;">FP16</td>
      <td align="right" style="border-top: 2px solid #666;"><strong>73.56781</strong></td>
      <td align="right" style="border-top: 2px solid #666;">47.91949 (65.14%)</td>
      <td align="right" style="border-top: 2px solid #666;"><strong>13.21552 (17.96%)</strong></td>
      <td align="right" style="border-top: 2px solid #666;"><strong>0.27591 (0.38%)</strong></td>
      <td align="right" style="border-top: 2px solid #666;"><strong>2.08314 (2.83%)</strong></td>
      <td align="right" style="border-top: 2px solid #666;">9.97480 (13.56%)</td>
    </tr>
    <tr>
      <td align="left">DiffKV</td>
      <td align="right">105.81041</td>
      <td align="right">48.03355 (45.40%)</td>
      <td align="right">18.44208 (17.43%)</td>
      <td align="right">2.78817 (2.64%)</td>
      <td align="right">2.25281 (2.13%)</td>
      <td align="right">34.13334 (32.26%)</td>
    </tr>
    <tr>
      <td align="left">DiffKV-v1</td>
      <td align="right">77.05523</td>
      <td align="right">48.02847 (62.33%)</td>
      <td align="right">18.43469 (23.92%)</td>
      <td align="right">2.78872 (3.62%)</td>
      <td align="right">2.26674 (2.94%)</td>
      <td align="right"><strong>5.44613 (7.07%)</strong></td>
    </tr>
    <tr>
      <td rowspan="3" align="right" valign="middle" style="border-top: 2px solid #666;">16</td>
      <td rowspan="3" align="right" valign="middle" style="border-top: 2px solid #666;">512</td>
      <td rowspan="3" align="right" valign="middle" style="border-top: 2px solid #666;">512</td>
      <td align="left" style="border-top: 2px solid #666;">FP16</td>
      <td align="right" style="border-top: 2px solid #666;"><strong>17.96300</strong></td>
      <td align="right" style="border-top: 2px solid #666;">13.72659 (76.42%)</td>
      <td align="right" style="border-top: 2px solid #666;"><strong>1.65576 (9.22%)</strong></td>
      <td align="right" style="border-top: 2px solid #666;"><strong>0.07224 (0.40%)</strong></td>
      <td align="right" style="border-top: 2px solid #666;"><strong>0.57717 (3.21%)</strong></td>
      <td align="right" style="border-top: 2px solid #666;">1.91297 (10.65%)</td>
    </tr>
    <tr>
      <td align="left">DiffKV</td>
      <td align="right">24.25620</td>
      <td align="right">13.75005 (56.69%)</td>
      <td align="right">2.89309 (11.93%)</td>
      <td align="right">0.55097 (2.27%)</td>
      <td align="right">0.62339 (2.57%)</td>
      <td align="right">6.40141 (26.39%)</td>
    </tr>
    <tr>
      <td align="left">DiffKV-v1</td>
      <td align="right">19.38419</td>
      <td align="right">13.74832 (70.93%)</td>
      <td align="right">2.89073 (14.91%)</td>
      <td align="right">0.55065 (2.84%)</td>
      <td align="right">0.62462 (3.22%)</td>
      <td align="right"><strong>1.55125 (8.00%)</strong></td>
    </tr>
    <tr>
      <td rowspan="3" align="right" valign="middle" style="border-top: 2px solid #666;">16</td>
      <td rowspan="3" align="right" valign="middle" style="border-top: 2px solid #666;">512</td>
      <td rowspan="3" align="right" valign="middle" style="border-top: 2px solid #666;">1024</td>
      <td align="left" style="border-top: 2px solid #666;">FP16</td>
      <td align="right" style="border-top: 2px solid #666;"><strong>35.33025</strong></td>
      <td align="right" style="border-top: 2px solid #666;">25.47088 (72.09%)</td>
      <td align="right" style="border-top: 2px solid #666;"><strong>4.36293 (12.35%)</strong></td>
      <td align="right" style="border-top: 2px solid #666;"><strong>0.13642 (0.39%)</strong></td>
      <td align="right" style="border-top: 2px solid #666;"><strong>1.08870 (3.08%)</strong></td>
      <td align="right" style="border-top: 2px solid #666;">4.23013 (11.97%)</td>
    </tr>
    <tr>
      <td align="left">DiffKV</td>
      <td align="right">51.42691</td>
      <td align="right">25.52429 (49.63%)</td>
      <td align="right">7.08392 (13.77%)</td>
      <td align="right">1.12431 (2.19%)</td>
      <td align="right">1.17076 (2.28%)</td>
      <td align="right">16.44822 (31.98%)</td>
    </tr>
    <tr>
      <td align="left">DiffKV-v1</td>
      <td align="right">38.02322</td>
      <td align="right">25.52196 (67.12%)</td>
      <td align="right">7.07548 (18.61%)</td>
      <td align="right">1.12368 (2.96%)</td>
      <td align="right">1.17726 (3.10%)</td>
      <td align="right"><strong>3.08506 (8.11%)</strong></td>
    </tr>
    <tr>
      <td rowspan="3" align="right" valign="middle" style="border-top: 2px solid #666;">16</td>
      <td rowspan="3" align="right" valign="middle" style="border-top: 2px solid #666;">1024</td>
      <td rowspan="3" align="right" valign="middle" style="border-top: 2px solid #666;">512</td>
      <td align="left" style="border-top: 2px solid #666;">FP16</td>
      <td align="right" style="border-top: 2px solid #666;"><strong>21.79236</strong></td>
      <td align="right" style="border-top: 2px solid #666;">15.73119 (72.19%)</td>
      <td align="right" style="border-top: 2px solid #666;"><strong>2.79546 (12.83%)</strong></td>
      <td align="right" style="border-top: 2px solid #666;"><strong>0.08034 (0.37%)</strong></td>
      <td align="right" style="border-top: 2px solid #666;"><strong>0.64520 (2.96%)</strong></td>
      <td align="right" style="border-top: 2px solid #666;">2.51703 (11.55%)</td>
    </tr>
    <tr>
      <td align="left">DiffKV</td>
      <td align="right">28.42568</td>
      <td align="right">15.76153 (55.45%)</td>
      <td align="right">4.21689 (14.83%)</td>
      <td align="right">0.59810 (2.10%)</td>
      <td align="right">0.69584 (2.45%)</td>
      <td align="right">7.11451 (25.03%)</td>
    </tr>
    <tr>
      <td align="left">DiffKV-v1</td>
      <td align="right">23.14264</td>
      <td align="right">15.76005 (68.10%)</td>
      <td align="right">4.21351 (18.21%)</td>
      <td align="right">0.59762 (2.58%)</td>
      <td align="right">0.69747 (3.01%)</td>
      <td align="right"><strong>1.85235 (8.00%)</strong></td>
    </tr>
    <tr>
      <td rowspan="3" align="right" valign="middle" style="border-top: 2px solid #666;">16</td>
      <td rowspan="3" align="right" valign="middle" style="border-top: 2px solid #666;">1024</td>
      <td rowspan="3" align="right" valign="middle" style="border-top: 2px solid #666;">1024</td>
      <td align="left" style="border-top: 2px solid #666;">FP16</td>
      <td align="right" style="border-top: 2px solid #666;"><strong>40.49930</strong></td>
      <td align="right" style="border-top: 2px solid #666;">27.47579 (67.84%)</td>
      <td align="right" style="border-top: 2px solid #666;"><strong>6.59216 (16.28%)</strong></td>
      <td align="right" style="border-top: 2px solid #666;"><strong>0.14450 (0.36%)</strong></td>
      <td align="right" style="border-top: 2px solid #666;"><strong>1.15616 (2.85%)</strong></td>
      <td align="right" style="border-top: 2px solid #666;">5.08027 (12.54%)</td>
    </tr>
    <tr>
      <td align="left">DiffKV</td>
      <td align="right">53.82288</td>
      <td align="right">27.53895 (51.17%)</td>
      <td align="right">9.60579 (17.85%)</td>
      <td align="right">1.20764 (2.24%)</td>
      <td align="right">1.24781 (2.32%)</td>
      <td align="right">14.14245 (26.28%)</td>
    </tr>
    <tr>
      <td align="left">DiffKV-v1</td>
      <td align="right">42.79380</td>
      <td align="right">27.53750 (64.35%)</td>
      <td align="right">9.60063 (22.43%)</td>
      <td align="right">1.20701 (2.82%)</td>
      <td align="right">1.25053 (2.92%)</td>
      <td align="right"><strong>3.15266 (7.37%)</strong></td>
    </tr>
    <tr>
      <td rowspan="3" align="right" valign="middle" style="border-top: 2px solid #666;">32</td>
      <td rowspan="3" align="right" valign="middle" style="border-top: 2px solid #666;">512</td>
      <td rowspan="3" align="right" valign="middle" style="border-top: 2px solid #666;">512</td>
      <td align="left" style="border-top: 2px solid #666;">FP16</td>
      <td align="right" style="border-top: 2px solid #666;"><strong>22.80581</strong></td>
      <td align="right" style="border-top: 2px solid #666;">15.59297 (68.37%)</td>
      <td align="right" style="border-top: 2px solid #666;"><strong>3.18377 (13.96%)</strong></td>
      <td align="right" style="border-top: 2px solid #666;"><strong>0.08172 (0.36%)</strong></td>
      <td align="right" style="border-top: 2px solid #666;"><strong>0.70099 (3.07%)</strong></td>
      <td align="right" style="border-top: 2px solid #666;">3.20445 (14.05%)</td>
    </tr>
    <tr>
      <td align="left">DiffKV</td>
      <td align="right">29.98198</td>
      <td align="right">15.60857 (52.06%)</td>
      <td align="right">5.08138 (16.95%)</td>
      <td align="right">0.56103 (1.87%)</td>
      <td align="right">0.75556 (2.52%)</td>
      <td align="right">7.92520 (26.43%)</td>
    </tr>
    <tr>
      <td align="left">DiffKV-v1</td>
      <td align="right">24.13988</td>
      <td align="right">15.60637 (64.65%)</td>
      <td align="right">5.07723 (21.03%)</td>
      <td align="right">0.56037 (2.32%)</td>
      <td align="right">0.75603 (3.13%)</td>
      <td align="right"><strong>2.10001 (8.70%)</strong></td>
    </tr>
  </tbody>
</table>

동일 조건과 항목에서 최솟값이 최댓값보다 2%를 초과해 작으면 최솟값을 굵게 표시했습니다.

원본 표: profiling-TOBE-table.md, profiling-DIffKV-table.md.
Others = CPU ops, scheduling, idle 합산
