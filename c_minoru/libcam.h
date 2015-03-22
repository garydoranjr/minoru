/*
 * Copyright (C) 2009 Giacomo Spigler
 * CopyPolicy: Released under the terms of the GNU GPL v3.0.
 */

#ifndef __LIBCAM__H__
#define __LIBCAM__H__


struct buffer {
        void *                  start;
        size_t                  length;
};





class Camera {
private:
  void Open();
  void Close();

  void Init();
  void UnInit();

  void Start();
  void Stop();

  void init_mmap();

  bool initialised;


public:
  const char *name;  //dev_name
  int width;
  int height;
  int fps;

  int w2;

  unsigned char *data;

  int fd;
  buffer *buffers;
  int n_buffers;

  Camera(const char *name, int w, int h, int fps=30);
  ~Camera();

  unsigned char *Get();    //deprecated
  bool Update(unsigned int t=100, int timeout_ms=500); //better  (t=0.1ms, in usecs)
  bool Update(Camera *c2, unsigned int t=100, int timeout_ms=500);
  void toArray(unsigned char *_l);
  void StopCam();

};





#endif
