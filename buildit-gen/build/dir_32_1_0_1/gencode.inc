#define A_dim (512)
#define B_dim (512)
#define C_dim (196)
#define Gy_i (0)
#define Gy_d (1)
#define A_blocks (32)
#define C_blocks (1)
#define offsets_size (32768)
float __device__ max_f(float a, float b) {return a>b?a:b;}
void __global__ mm(const float * __restrict__ BC, const sparse_matrix AB, const float * __restrict__ bias, float *AC) {
  register float ACC[21] = {0.0}; 
{
    float var0 = 0.0f;
    int var1 = (blockIdx.y * 196) + threadIdx.y;
    int var2 = (blockIdx.x * 2) + threadIdx.x;
    if (0 == var2) {
    } else {
      if (1 == var2) {
        for (int var3 = 0; var3 < 18; var3 = var3 + 1) {
          float var4 = 0.0f;
          for (int var5 = AB.rows[var3]; var5 < AB.rows[var3 + 1]; var5 = var5 + 1) {
            int var6 = AB.column_val[var5];
            var4 = var4 + (BC[(var6 * 196) + var1] * AB.values_column[var5]);
          }
          AC[(var3 * 196) + var1] = max_f(var4 + bias[var3], 0.0f);
        }
      } else {
        if (2 == var2) {
        } else {
          if (3 == var2) {
            for (int var7 = 18; var7 < 33; var7 = var7 + 1) {
              float var8 = 0.0f;
              for (int var9 = AB.rows[var7]; var9 < AB.rows[var7 + 1]; var9 = var9 + 1) {
                int var10 = AB.column_val[var9];
                var8 = var8 + (BC[(var10 * 196) + var1] * AB.values_column[var9]);
              }
              AC[(var7 * 196) + var1] = max_f(var8 + bias[var7], 0.0f);
            }
          } else {
            if (4 == var2) {
            } else {
              if (5 == var2) {
                for (int var11 = 33; var11 < 46; var11 = var11 + 1) {
                  float var12 = 0.0f;
                  for (int var13 = AB.rows[var11]; var13 < AB.rows[var11 + 1]; var13 = var13 + 1) {
                    int var14 = AB.column_val[var13];
                    var12 = var12 + (BC[(var14 * 196) + var1] * AB.values_column[var13]);
                  }
                  AC[(var11 * 196) + var1] = max_f(var12 + bias[var11], 0.0f);
                }
              } else {
                if (6 == var2) {
                } else {
                  if (7 == var2) {
                    for (int var15 = 46; var15 < 64; var15 = var15 + 1) {
                      float var16 = 0.0f;
                      for (int var17 = AB.rows[var15]; var17 < AB.rows[var15 + 1]; var17 = var17 + 1) {
                        int var18 = AB.column_val[var17];
                        var16 = var16 + (BC[(var18 * 196) + var1] * AB.values_column[var17]);
                      }
                      AC[(var15 * 196) + var1] = max_f(var16 + bias[var15], 0.0f);
                    }
                  } else {
                    if (8 == var2) {
                    } else {
                      if (9 == var2) {
                        for (int var19 = 64; var19 < 78; var19 = var19 + 1) {
                          float var20 = 0.0f;
                          for (int var21 = AB.rows[var19]; var21 < AB.rows[var19 + 1]; var21 = var21 + 1) {
                            int var22 = AB.column_val[var21];
                            var20 = var20 + (BC[(var22 * 196) + var1] * AB.values_column[var21]);
                          }
                          AC[(var19 * 196) + var1] = max_f(var20 + bias[var19], 0.0f);
                        }
                      } else {
                        if (10 == var2) {
                        } else {
                          if (11 == var2) {
                            for (int var23 = 78; var23 < 92; var23 = var23 + 1) {
                              float var24 = 0.0f;
                              for (int var25 = AB.rows[var23]; var25 < AB.rows[var23 + 1]; var25 = var25 + 1) {
                                int var26 = AB.column_val[var25];
                                var24 = var24 + (BC[(var26 * 196) + var1] * AB.values_column[var25]);
                              }
                              AC[(var23 * 196) + var1] = max_f(var24 + bias[var23], 0.0f);
                            }
                          } else {
                            if (12 == var2) {
                            } else {
                              if (13 == var2) {
                                for (int var27 = 92; var27 < 108; var27 = var27 + 1) {
                                  float var28 = 0.0f;
                                  for (int var29 = AB.rows[var27]; var29 < AB.rows[var27 + 1]; var29 = var29 + 1) {
                                    int var30 = AB.column_val[var29];
                                    var28 = var28 + (BC[(var30 * 196) + var1] * AB.values_column[var29]);
                                  }
                                  AC[(var27 * 196) + var1] = max_f(var28 + bias[var27], 0.0f);
                                }
                              } else {
                                if (14 == var2) {
                                } else {
                                  if (15 == var2) {
                                    for (int var31 = 108; var31 < 128; var31 = var31 + 1) {
                                      float var32 = 0.0f;
                                      for (int var33 = AB.rows[var31]; var33 < AB.rows[var31 + 1]; var33 = var33 + 1) {
                                        int var34 = AB.column_val[var33];
                                        var32 = var32 + (BC[(var34 * 196) + var1] * AB.values_column[var33]);
                                      }
                                      AC[(var31 * 196) + var1] = max_f(var32 + bias[var31], 0.0f);
                                    }
                                  } else {
                                    if (16 == var2) {
                                    } else {
                                      if (17 == var2) {
                                        for (int var35 = 128; var35 < 141; var35 = var35 + 1) {
                                          float var36 = 0.0f;
                                          for (int var37 = AB.rows[var35]; var37 < AB.rows[var35 + 1]; var37 = var37 + 1) {
                                            int var38 = AB.column_val[var37];
                                            var36 = var36 + (BC[(var38 * 196) + var1] * AB.values_column[var37]);
                                          }
                                          AC[(var35 * 196) + var1] = max_f(var36 + bias[var35], 0.0f);
                                        }
                                      } else {
                                        if (18 == var2) {
                                        } else {
                                          if (19 == var2) {
                                            for (int var39 = 141; var39 < 155; var39 = var39 + 1) {
                                              float var40 = 0.0f;
                                              for (int var41 = AB.rows[var39]; var41 < AB.rows[var39 + 1]; var41 = var41 + 1) {
                                                int var42 = AB.column_val[var41];
                                                var40 = var40 + (BC[(var42 * 196) + var1] * AB.values_column[var41]);
                                              }
                                              AC[(var39 * 196) + var1] = max_f(var40 + bias[var39], 0.0f);
                                            }
                                          } else {
                                            if (20 == var2) {
                                            } else {
                                              if (21 == var2) {
                                                for (int var43 = 155; var43 < 173; var43 = var43 + 1) {
                                                  float var44 = 0.0f;
                                                  for (int var45 = AB.rows[var43]; var45 < AB.rows[var43 + 1]; var45 = var45 + 1) {
                                                    int var46 = AB.column_val[var45];
                                                    var44 = var44 + (BC[(var46 * 196) + var1] * AB.values_column[var45]);
                                                  }
                                                  AC[(var43 * 196) + var1] = max_f(var44 + bias[var43], 0.0f);
                                                }
                                              } else {
                                                if (22 == var2) {
                                                } else {
                                                  if (23 == var2) {
                                                    for (int var47 = 173; var47 < 185; var47 = var47 + 1) {
                                                      float var48 = 0.0f;
                                                      for (int var49 = AB.rows[var47]; var49 < AB.rows[var47 + 1]; var49 = var49 + 1) {
                                                        int var50 = AB.column_val[var49];
                                                        var48 = var48 + (BC[(var50 * 196) + var1] * AB.values_column[var49]);
                                                      }
                                                      AC[(var47 * 196) + var1] = max_f(var48 + bias[var47], 0.0f);
                                                    }
                                                  } else {
                                                    if (24 == var2) {
                                                    } else {
                                                      if (25 == var2) {
                                                        for (int var51 = 185; var51 < 200; var51 = var51 + 1) {
                                                          float var52 = 0.0f;
                                                          for (int var53 = AB.rows[var51]; var53 < AB.rows[var51 + 1]; var53 = var53 + 1) {
                                                            int var54 = AB.column_val[var53];
                                                            var52 = var52 + (BC[(var54 * 196) + var1] * AB.values_column[var53]);
                                                          }
                                                          AC[(var51 * 196) + var1] = max_f(var52 + bias[var51], 0.0f);
                                                        }
                                                      } else {
                                                        if (26 == var2) {
                                                        } else {
                                                          if (27 == var2) {
                                                            for (int var55 = 200; var55 < 218; var55 = var55 + 1) {
                                                              float var56 = 0.0f;
                                                              for (int var57 = AB.rows[var55]; var57 < AB.rows[var55 + 1]; var57 = var57 + 1) {
                                                                int var58 = AB.column_val[var57];
                                                                var56 = var56 + (BC[(var58 * 196) + var1] * AB.values_column[var57]);
                                                              }
                                                              AC[(var55 * 196) + var1] = max_f(var56 + bias[var55], 0.0f);
                                                            }
                                                          } else {
                                                            if (28 == var2) {
                                                            } else {
                                                              if (29 == var2) {
                                                                for (int var59 = 218; var59 < 236; var59 = var59 + 1) {
                                                                  float var60 = 0.0f;
                                                                  for (int var61 = AB.rows[var59]; var61 < AB.rows[var59 + 1]; var61 = var61 + 1) {
                                                                    int var62 = AB.column_val[var61];
                                                                    var60 = var60 + (BC[(var62 * 196) + var1] * AB.values_column[var61]);
                                                                  }
                                                                  AC[(var59 * 196) + var1] = max_f(var60 + bias[var59], 0.0f);
                                                                }
                                                              } else {
                                                                if (30 == var2) {
                                                                } else {
                                                                  if (31 == var2) {
                                                                    for (int var63 = 236; var63 < 251; var63 = var63 + 1) {
                                                                      float var64 = 0.0f;
                                                                      for (int var65 = AB.rows[var63]; var65 < AB.rows[var63 + 1]; var65 = var65 + 1) {
                                                                        int var66 = AB.column_val[var65];
                                                                        var64 = var64 + (BC[(var66 * 196) + var1] * AB.values_column[var65]);
                                                                      }
                                                                      AC[(var63 * 196) + var1] = max_f(var64 + bias[var63], 0.0f);
                                                                    }
                                                                  } else {
                                                                    if (32 == var2) {
                                                                    } else {
                                                                      if (33 == var2) {
                                                                        for (int var67 = 251; var67 < 266; var67 = var67 + 1) {
                                                                          float var68 = 0.0f;
                                                                          for (int var69 = AB.rows[var67]; var69 < AB.rows[var67 + 1]; var69 = var69 + 1) {
                                                                            int var70 = AB.column_val[var69];
                                                                            var68 = var68 + (BC[(var70 * 196) + var1] * AB.values_column[var69]);
                                                                          }
                                                                          AC[(var67 * 196) + var1] = max_f(var68 + bias[var67], 0.0f);
                                                                        }
                                                                      } else {
                                                                        if (34 == var2) {
                                                                        } else {
                                                                          if (35 == var2) {
                                                                            for (int var71 = 266; var71 < 284; var71 = var71 + 1) {
                                                                              float var72 = 0.0f;
                                                                              for (int var73 = AB.rows[var71]; var73 < AB.rows[var71 + 1]; var73 = var73 + 1) {
                                                                                int var74 = AB.column_val[var73];
                                                                                var72 = var72 + (BC[(var74 * 196) + var1] * AB.values_column[var73]);
                                                                              }
                                                                              AC[(var71 * 196) + var1] = max_f(var72 + bias[var71], 0.0f);
                                                                            }
                                                                          } else {
                                                                            if (36 == var2) {
                                                                            } else {
                                                                              if (37 == var2) {
                                                                                for (int var75 = 284; var75 < 300; var75 = var75 + 1) {
                                                                                  float var76 = 0.0f;
                                                                                  for (int var77 = AB.rows[var75]; var77 < AB.rows[var75 + 1]; var77 = var77 + 1) {
                                                                                    int var78 = AB.column_val[var77];
                                                                                    var76 = var76 + (BC[(var78 * 196) + var1] * AB.values_column[var77]);
                                                                                  }
                                                                                  AC[(var75 * 196) + var1] = max_f(var76 + bias[var75], 0.0f);
                                                                                }
                                                                              } else {
                                                                                if (38 == var2) {
                                                                                } else {
                                                                                  if (39 == var2) {
                                                                                    for (int var79 = 300; var79 < 315; var79 = var79 + 1) {
                                                                                      float var80 = 0.0f;
                                                                                      for (int var81 = AB.rows[var79]; var81 < AB.rows[var79 + 1]; var81 = var81 + 1) {
                                                                                        int var82 = AB.column_val[var81];
                                                                                        var80 = var80 + (BC[(var82 * 196) + var1] * AB.values_column[var81]);
                                                                                      }
                                                                                      AC[(var79 * 196) + var1] = max_f(var80 + bias[var79], 0.0f);
                                                                                    }
                                                                                  } else {
                                                                                    if (40 == var2) {
                                                                                    } else {
                                                                                      if (41 == var2) {
                                                                                        for (int var83 = 315; var83 < 332; var83 = var83 + 1) {
                                                                                          float var84 = 0.0f;
                                                                                          for (int var85 = AB.rows[var83]; var85 < AB.rows[var83 + 1]; var85 = var85 + 1) {
                                                                                            int var86 = AB.column_val[var85];
                                                                                            var84 = var84 + (BC[(var86 * 196) + var1] * AB.values_column[var85]);
                                                                                          }
                                                                                          AC[(var83 * 196) + var1] = max_f(var84 + bias[var83], 0.0f);
                                                                                        }
                                                                                      } else {
                                                                                        if (42 == var2) {
                                                                                        } else {
                                                                                          if (43 == var2) {
                                                                                            for (int var87 = 332; var87 < 353; var87 = var87 + 1) {
                                                                                              float var88 = 0.0f;
                                                                                              for (int var89 = AB.rows[var87]; var89 < AB.rows[var87 + 1]; var89 = var89 + 1) {
                                                                                                int var90 = AB.column_val[var89];
                                                                                                var88 = var88 + (BC[(var90 * 196) + var1] * AB.values_column[var89]);
                                                                                              }
                                                                                              AC[(var87 * 196) + var1] = max_f(var88 + bias[var87], 0.0f);
                                                                                            }
                                                                                          } else {
                                                                                            if (44 == var2) {
                                                                                            } else {
                                                                                              if (45 == var2) {
                                                                                                for (int var91 = 353; var91 < 371; var91 = var91 + 1) {
                                                                                                  float var92 = 0.0f;
                                                                                                  for (int var93 = AB.rows[var91]; var93 < AB.rows[var91 + 1]; var93 = var93 + 1) {
                                                                                                    int var94 = AB.column_val[var93];
                                                                                                    var92 = var92 + (BC[(var94 * 196) + var1] * AB.values_column[var93]);
                                                                                                  }
                                                                                                  AC[(var91 * 196) + var1] = max_f(var92 + bias[var91], 0.0f);
                                                                                                }
                                                                                              } else {
                                                                                                if (46 == var2) {
                                                                                                } else {
                                                                                                  if (47 == var2) {
                                                                                                    for (int var95 = 371; var95 < 386; var95 = var95 + 1) {
                                                                                                      float var96 = 0.0f;
                                                                                                      for (int var97 = AB.rows[var95]; var97 < AB.rows[var95 + 1]; var97 = var97 + 1) {
                                                                                                        int var98 = AB.column_val[var97];
                                                                                                        var96 = var96 + (BC[(var98 * 196) + var1] * AB.values_column[var97]);
                                                                                                      }
                                                                                                      AC[(var95 * 196) + var1] = max_f(var96 + bias[var95], 0.0f);
                                                                                                    }
                                                                                                  } else {
                                                                                                    if (48 == var2) {
                                                                                                    } else {
                                                                                                      if (49 == var2) {
                                                                                                        for (int var99 = 386; var99 < 403; var99 = var99 + 1) {
                                                                                                          float var100 = 0.0f;
                                                                                                          for (int var101 = AB.rows[var99]; var101 < AB.rows[var99 + 1]; var101 = var101 + 1) {
                                                                                                            int var102 = AB.column_val[var101];
                                                                                                            var100 = var100 + (BC[(var102 * 196) + var1] * AB.values_column[var101]);
                                                                                                          }
                                                                                                          AC[(var99 * 196) + var1] = max_f(var100 + bias[var99], 0.0f);
                                                                                                        }
                                                                                                      } else {
                                                                                                        if (50 == var2) {
                                                                                                        } else {
                                                                                                          if (51 == var2) {
                                                                                                            for (int var103 = 403; var103 < 416; var103 = var103 + 1) {
                                                                                                              float var104 = 0.0f;
                                                                                                              for (int var105 = AB.rows[var103]; var105 < AB.rows[var103 + 1]; var105 = var105 + 1) {
                                                                                                                int var106 = AB.column_val[var105];
                                                                                                                var104 = var104 + (BC[(var106 * 196) + var1] * AB.values_column[var105]);
                                                                                                              }
                                                                                                              AC[(var103 * 196) + var1] = max_f(var104 + bias[var103], 0.0f);
                                                                                                            }
                                                                                                          } else {
                                                                                                            if (52 == var2) {
                                                                                                            } else {
                                                                                                              if (53 == var2) {
                                                                                                                for (int var107 = 416; var107 < 432; var107 = var107 + 1) {
                                                                                                                  float var108 = 0.0f;
                                                                                                                  for (int var109 = AB.rows[var107]; var109 < AB.rows[var107 + 1]; var109 = var109 + 1) {
                                                                                                                    int var110 = AB.column_val[var109];
                                                                                                                    var108 = var108 + (BC[(var110 * 196) + var1] * AB.values_column[var109]);
                                                                                                                  }
                                                                                                                  AC[(var107 * 196) + var1] = max_f(var108 + bias[var107], 0.0f);
                                                                                                                }
                                                                                                              } else {
                                                                                                                if (54 == var2) {
                                                                                                                } else {
                                                                                                                  if (55 == var2) {
                                                                                                                    for (int var111 = 432; var111 < 446; var111 = var111 + 1) {
                                                                                                                      float var112 = 0.0f;
                                                                                                                      for (int var113 = AB.rows[var111]; var113 < AB.rows[var111 + 1]; var113 = var113 + 1) {
                                                                                                                        int var114 = AB.column_val[var113];
                                                                                                                        var112 = var112 + (BC[(var114 * 196) + var1] * AB.values_column[var113]);
                                                                                                                      }
                                                                                                                      AC[(var111 * 196) + var1] = max_f(var112 + bias[var111], 0.0f);
                                                                                                                    }
                                                                                                                  } else {
                                                                                                                    if (56 == var2) {
                                                                                                                    } else {
                                                                                                                      if (57 == var2) {
                                                                                                                        for (int var115 = 446; var115 < 464; var115 = var115 + 1) {
                                                                                                                          float var116 = 0.0f;
                                                                                                                          for (int var117 = AB.rows[var115]; var117 < AB.rows[var115 + 1]; var117 = var117 + 1) {
                                                                                                                            int var118 = AB.column_val[var117];
                                                                                                                            var116 = var116 + (BC[(var118 * 196) + var1] * AB.values_column[var117]);
                                                                                                                          }
                                                                                                                          AC[(var115 * 196) + var1] = max_f(var116 + bias[var115], 0.0f);
                                                                                                                        }
                                                                                                                      } else {
                                                                                                                        if (58 == var2) {
                                                                                                                        } else {
                                                                                                                          if (59 == var2) {
                                                                                                                            for (int var119 = 464; var119 < 481; var119 = var119 + 1) {
                                                                                                                              float var120 = 0.0f;
                                                                                                                              for (int var121 = AB.rows[var119]; var121 < AB.rows[var119 + 1]; var121 = var121 + 1) {
                                                                                                                                int var122 = AB.column_val[var121];
                                                                                                                                var120 = var120 + (BC[(var122 * 196) + var1] * AB.values_column[var121]);
                                                                                                                              }
                                                                                                                              AC[(var119 * 196) + var1] = max_f(var120 + bias[var119], 0.0f);
                                                                                                                            }
                                                                                                                          } else {
                                                                                                                            if (60 == var2) {
                                                                                                                            } else {
                                                                                                                              if (61 == var2) {
                                                                                                                                for (int var123 = 481; var123 < 497; var123 = var123 + 1) {
                                                                                                                                  float var124 = 0.0f;
                                                                                                                                  for (int var125 = AB.rows[var123]; var125 < AB.rows[var123 + 1]; var125 = var125 + 1) {
                                                                                                                                    int var126 = AB.column_val[var125];
                                                                                                                                    var124 = var124 + (BC[(var126 * 196) + var1] * AB.values_column[var125]);
                                                                                                                                  }
                                                                                                                                  AC[(var123 * 196) + var1] = max_f(var124 + bias[var123], 0.0f);
                                                                                                                                }
                                                                                                                              } else {
                                                                                                                                if (62 == var2) {
                                                                                                                                } else {
                                                                                                                                  for (int var127 = 497; var127 < 512; var127 = var127 + 1) {
                                                                                                                                    float var128 = 0.0f;
                                                                                                                                    for (int var129 = AB.rows[var127]; var129 < AB.rows[var127 + 1]; var129 = var129 + 1) {
                                                                                                                                      int var130 = AB.column_val[var129];
                                                                                                                                      var128 = var128 + (BC[(var130 * 196) + var1] * AB.values_column[var129]);
                                                                                                                                    }
                                                                                                                                    AC[(var127 * 196) + var1] = max_f(var128 + bias[var127], 0.0f);
                                                                                                                                  }
                                                                                                                                }
                                                                                                                              }
                                                                                                                            }
                                                                                                                          }
                                                                                                                        }
                                                                                                                      }
                                                                                                                    }
                                                                                                                  }
                                                                                                                }
                                                                                                              }
                                                                                                            }
                                                                                                          }
                                                                                                        }
                                                                                                      }
                                                                                                    }
                                                                                                  }
                                                                                                }
                                                                                              }
                                                                                            }
                                                                                          }
                                                                                        }
                                                                                      }
                                                                                    }
                                                                                  }
                                                                                }
                                                                              }
                                                                            }
                                                                          }
                                                                        }
                                                                      }
                                                                    }
                                                                  }
                                                                }
                                                              }
                                                            }
                                                          }
                                                        }
                                                      }
                                                    }
                                                  }
                                                }
                                              }
                                            }
                                          }
                                        }
                                      }
                                    }
                                  }
                                }
                              }
                            }
                          }
                        }
                      }
                    }
                  }
                }
              }
            }
          }
        }
      }
    }
  }
}
