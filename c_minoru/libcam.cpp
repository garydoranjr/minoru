/*
 Copyright (C) 2009 Giacomo Spigler

 This program is free software: you can redistribute it and/or modify
 it under the terms of the GNU General Public License as published by
 the Free Software Foundation, either version 3 of the License, or
 (at your option) any later version.

 This program is distributed in the hope that it will be useful,
 but WITHOUT ANY WARRANTY; without even the implied warranty of
 MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 GNU General Public License for more details.

 You should have received a copy of the GNU General Public License
 along with this program.  If not, see <http://www.gnu.org/licenses/>.
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <assert.h>

#include <getopt.h>             /* getopt_long() */

#include <fcntl.h>              /* low-level i/o */
#include <unistd.h>
#include <errno.h>
#include <malloc.h>
#include <sys/stat.h>
#include <sys/types.h>
#include <sys/time.h>
#include <sys/mman.h>
#include <sys/ioctl.h>

#include <asm/types.h>          /* for videodev2.h */

#include <linux/videodev2.h>

#define CLEAR(x) memset (&(x), 0, sizeof (x))



#include "libcam.h"

static void errno_exit (const char *           s)
{
        fprintf (stderr, "%s error %d, %s\n",
                 s, errno, strerror (errno));

        exit (EXIT_FAILURE);
}

static int xioctl(int fd, int request, void *arg)
{
        int r;

        do {
		r = ioctl (fd, request, arg);
	}
        while ((-1 == r) && (EINTR == errno));

        return r;
}

Camera::Camera(const char *n, int w, int h, int f) {
  name=n;
  width=w;
  height=h;
  fps=f;

  w2=w/2;

  data=(unsigned char *)malloc(w*h*4);

  this->Open();
  this->Init();
  this->Start();
  initialised = true;
}

Camera::~Camera() {
  this->StopCam();
}

void Camera::StopCam()
{
  if (initialised) {
    this->Stop();
    this->UnInit();
    this->Close();

    free(data);
    initialised = false;
  }
}

void Camera::Open() {
  struct stat st;
  if(-1==stat(name, &st)) {
    fprintf(stderr, "Cannot identify '%s' : %d, %s\n", name, errno, strerror(errno));
    exit(1);
  }

  if(!S_ISCHR(st.st_mode)) {
    fprintf(stderr, "%s is no device\n", name);
    exit(1);
  }

  fd=open(name, O_RDWR | O_NONBLOCK, 0);

  if(-1 == fd) {
    fprintf(stderr, "Cannot open '%s': %d, %s\n", name, errno, strerror(errno));
    exit(1);
  }

}

void Camera::Close() {
  if(-1==close(fd)) {
    errno_exit("close");
  }
  fd=-1;

}

void Camera::Init() {
  struct v4l2_capability cap;
  struct v4l2_cropcap cropcap;
  struct v4l2_crop crop;
  struct v4l2_format fmt;
  unsigned int min;

  if(-1 == xioctl (fd, VIDIOC_QUERYCAP, &cap)) {
    if (EINVAL == errno) {
      fprintf(stderr, "%s is no V4L2 device\n",name);
      exit(1);
    } else {
       errno_exit("VIDIOC_QUERYCAP");
    }
  }

  if(!(cap.capabilities & V4L2_CAP_VIDEO_CAPTURE)) {
    fprintf(stderr, "%s is no video capture device\n", name);
    exit(1);
  }

  if(!(cap.capabilities & V4L2_CAP_STREAMING)) {
    fprintf (stderr, "%s does not support streaming i/o\n", name);
    exit(1);
  }

  CLEAR (cropcap);

  cropcap.type = V4L2_BUF_TYPE_VIDEO_CAPTURE;

  if(0 == xioctl (fd, VIDIOC_CROPCAP, &cropcap)) {
    crop.type = V4L2_BUF_TYPE_VIDEO_CAPTURE;
    crop.c = cropcap.defrect; /* reset to default */

    if(-1 == xioctl (fd, VIDIOC_S_CROP, &crop)) {
      switch (errno) {
        case EINVAL:
          /* Cropping not supported. */
          break;
        default:
          /* Errors ignored. */
          break;
        }
      }
    } else {
      /* Errors ignored. */
    }

    CLEAR (fmt);

    fmt.type                = V4L2_BUF_TYPE_VIDEO_CAPTURE;
    fmt.fmt.pix.width       = width;
    fmt.fmt.pix.height      = height;
    fmt.fmt.pix.pixelformat = V4L2_PIX_FMT_YUYV;
    fmt.fmt.pix.field       = V4L2_FIELD_INTERLACED;


  if(-1 == xioctl (fd, VIDIOC_S_FMT, &fmt))
    errno_exit ("VIDIOC_S_FMT");



/*
struct v4l2_standard s;
s.name[0]='A';
s.frameperiod.numerator=1;
s.frameperiod.denominator=fps;

if(-1==xioctl(fd, VIDIOC_S_STD, &s))
  errno_exit("VIDIOC_S_STD");
*/


struct v4l2_streamparm p;
p.type=V4L2_BUF_TYPE_VIDEO_CAPTURE;
//p.parm.capture.capability=V4L2_CAP_TIMEPERFRAME;
//p.parm.capture.capturemode=V4L2_MODE_HIGHQUALITY;
p.parm.capture.timeperframe.numerator=1;
p.parm.capture.timeperframe.denominator=fps;
p.parm.output.timeperframe.numerator=1;
p.parm.output.timeperframe.denominator=fps;
//p.parm.output.outputmode=V4L2_MODE_HIGHQUALITY;
//p.parm.capture.extendedmode=0;
//p.parm.capture.readbuffers=n_buffers;


if(-1==xioctl(fd, VIDIOC_S_PARM, &p))
  errno_exit("VIDIOC_S_PARM");

  /* Note VIDIOC_S_FMT may change width and height. */
  /* Buggy driver paranoia. */
  min = fmt.fmt.pix.width * 2;
  if(fmt.fmt.pix.bytesperline < min)
    fmt.fmt.pix.bytesperline = min;
  min = fmt.fmt.pix.bytesperline * fmt.fmt.pix.height;
  if(fmt.fmt.pix.sizeimage < min)
    fmt.fmt.pix.sizeimage = min;

  init_mmap();

}

void Camera::init_mmap() {
  struct v4l2_requestbuffers req;

  CLEAR (req);

  req.count               = 4;
  req.type                = V4L2_BUF_TYPE_VIDEO_CAPTURE;
  req.memory              = V4L2_MEMORY_MMAP;

  if(-1 == xioctl (fd, VIDIOC_REQBUFS, &req)) {
    if(EINVAL == errno) {
      fprintf (stderr, "%s does not support memory mapping\n", name);
      exit (1);
    } else {
      errno_exit ("VIDIOC_REQBUFS");
    }
  }

  if(req.count < 2) {
    fprintf (stderr, "Insufficient buffer memory on %s\n", name);
    exit(1);
  }

  buffers = (buffer *)calloc(req.count, sizeof (*buffers));

  if(!buffers) {
    fprintf(stderr, "Out of memory\n");
    exit(1);
  }

  for(n_buffers = 0; n_buffers < (int)req.count; ++n_buffers) {
    struct v4l2_buffer buf;

    CLEAR (buf);

    buf.type        = V4L2_BUF_TYPE_VIDEO_CAPTURE;
    buf.memory      = V4L2_MEMORY_MMAP;
    buf.index       = n_buffers;

    if(-1 == xioctl (fd, VIDIOC_QUERYBUF, &buf))
      errno_exit ("VIDIOC_QUERYBUF");

    buffers[n_buffers].length = buf.length;
    buffers[n_buffers].start = mmap (NULL /* start anywhere */,
                              buf.length,
                              PROT_READ | PROT_WRITE /* required */,
                              MAP_SHARED /* recommended */,
                              fd, buf.m.offset);

    if(MAP_FAILED == buffers[n_buffers].start)
      errno_exit ("mmap");
  }

}

void Camera::UnInit() {
  unsigned int i;

  for(i = 0; i < (unsigned int)n_buffers; ++i) {
    if(-1 == munmap (buffers[i].start, buffers[i].length)) {
      errno_exit ("munmap");
    }
  }

  free (buffers);
}

void Camera::Start() {
  unsigned int i;
  enum v4l2_buf_type type;

  for(i = 0; i < (unsigned int)n_buffers; ++i) {
    struct v4l2_buffer buf;

    CLEAR (buf);

    buf.type        = V4L2_BUF_TYPE_VIDEO_CAPTURE;
    buf.memory      = V4L2_MEMORY_MMAP;
    buf.index       = i;

    if(-1 == xioctl (fd, VIDIOC_QBUF, &buf))
      errno_exit ("VIDIOC_QBUF");
  }

  type = V4L2_BUF_TYPE_VIDEO_CAPTURE;

  if(-1 == xioctl (fd, VIDIOC_STREAMON, &type)) {
    errno_exit ("VIDIOC_STREAMON");
  }


}

void Camera::Stop() {
  enum v4l2_buf_type type;

  type = V4L2_BUF_TYPE_VIDEO_CAPTURE;

  if(-1 == xioctl (fd, VIDIOC_STREAMOFF, &type)) {
    errno_exit ("VIDIOC_STREAMOFF");
  }

}

unsigned char *Camera::Get() {
  struct v4l2_buffer buf;
  CLEAR(buf);
  
  buf.type = V4L2_BUF_TYPE_VIDEO_CAPTURE;
  buf.memory = V4L2_MEMORY_MMAP;
  if(-1 == xioctl (fd, VIDIOC_DQBUF, &buf)) {
  switch (errno) {
    case EAGAIN:
      return 0;
    case EIO:
    default:
      return 0; //errno_exit ("VIDIOC_DQBUF");
  }
  }
  
  assert(buf.index < (unsigned int)n_buffers);
  
  memcpy(data, (unsigned char *)buffers[buf.index].start, buffers[buf.index].length);
  
  if(-1 == xioctl (fd, VIDIOC_QBUF, &buf))
  return 0; //errno_exit ("VIDIOC_QBUF");
  
  return data;
}

bool Camera::Update(unsigned int t, int timeout_ms) {
  bool grabbed = false;
  int grab_time_uS = 0;
  while (!grabbed) {
    if ((!grabbed) && (this->Get()!=0)) grabbed = true;
    if (!grabbed) {
      usleep(t);
      grab_time_uS+=(int)t;
      if (grab_time_uS > timeout_ms * 1000) {
        break;
      }
    }
  }

  return grabbed;

}

bool Camera::Update(Camera *c2, unsigned int t, int timeout_ms) {
  bool left_grabbed = false;
  bool right_grabbed = false;
  int grab_time_uS = 0;
  while (!(left_grabbed && right_grabbed)) {
    if ((!left_grabbed) && (this->Get()!=0)) left_grabbed = true;
    if ((!right_grabbed) && (c2->Get()!=0)) right_grabbed = true;
    if (!(left_grabbed && right_grabbed)) {
      usleep(t);
      grab_time_uS+=(int)t;
      if (grab_time_uS > timeout_ms * 1000) {
        break;
      }
    }
  }

  return left_grabbed & right_grabbed;

}

void Camera::toArray(unsigned char *l_) {

  for(int x=0; x<w2; x++) {
    for(int y=0; y<height; y++) {
      int y0, y1, u, v; //y0 u y1 v

      int i=(y*w2+x)*4;
      y0=data[i];
      u=data[i+1];
      y1=data[i+2];
      v=data[i+3];

      int r, g, b;
      r = y0 + (1.370705 * (v-128));
      g = y0 - (0.698001 * (v-128)) - (0.337633 * (u-128));
      b = y0 + (1.732446 * (u-128));

      if(r > 255) r = 255;
      if(g > 255) g = 255;
      if(b > 255) b = 255;
      if(r < 0) r = 0;
      if(g < 0) g = 0;
      if(b < 0) b = 0;

      i=(y*width+2*x)*3;
      l_[i] = (unsigned char)(r); //R
      l_[i+1] = (unsigned char)(g); //G
      l_[i+2] = (unsigned char)(b); //B


      r = y1 + (1.370705 * (v-128));
      g = y1 - (0.698001 * (v-128)) - (0.337633 * (u-128));
      b = y1 + (1.732446 * (u-128));

      if(r > 255) r = 255;
      if(g > 255) g = 255;
      if(b > 255) b = 255;
      if(r < 0) r = 0;
      if(g < 0) g = 0;
      if(b < 0) b = 0;

      l_[i+3] = (unsigned char)(r); //R
      l_[i+4] = (unsigned char)(g); //G
      l_[i+5] = (unsigned char)(b); //B

    }
  }

}
