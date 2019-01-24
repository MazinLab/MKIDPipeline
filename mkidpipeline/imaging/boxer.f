      SUBROUTINE BOXER (IS, JS, X, Y, DAREA)
CF2PY INTENT(OUT) :: DAREA
CF2PY INTEGER :: IS
CF2PY INTEGER :: JS
CF2PY DOUBLE :: X(4)
CF2PY DOUBLE :: Y(4)
C
C
C--------------------------------------------------------------------------
C Cut out of file 'drutil.f' from the pydrizzle package in the
C stsci_python distribution (v. 2.13).
C Note - F2PY directives above added by JvE, 7/2/2013, so that F2PY compiles
C properly. To compile on any platform, in theory, type at the shell
C command line:
C
C   f2py -c -m boxer boxer.f
C
C This should make an output file 'boxer.so' (on MacOS) or something similar,
C which can be imported into Python with 'import boxer'.
C
C To call boxer from python, use e.g.:
C
C       from photonlist.boxer import boxer
C       area = boxer(is, js, x, y)
C
C where:
C       is, js = integers, coordinates of *center* of a unit square
C       x, y = 4-element vectors of the x and y coordinates of the
C              corners of a quadrilateral.
C Returns the area of the unit square which overlaps with the quadrilateral.
C JvE 7/2/2013
C---------------------------------------------------------------------------
C
C BOXER -- compute area of box overlap
C
C Calculate the area common to input clockwise polygon x(n), y(n) with
C square (is, js) to (is+1, js+1).
C This version is for a quadrilateral.
C
C W.B. Sparks STScI 2-June-1990.
C Phil Hodge        20-Nov-1990  Change calling sequence; single precision.
C Richard Hook ECF  24-Apr-1996  Change coordinate origin
C                                so that centre of pixel has integer position
C                   03-Jan-2001  Removed accuracy check

      IMPLICIT NONE

      INTEGER IS, JS
      DOUBLE PRECISION X(*), Y(*)
      DOUBLE PRECISION DAREA
C--
      DOUBLE PRECISION PX(4), PY(4), SUM
      DOUBLE PRECISION SGAREA
      INTEGER I

C Set up coords relative to unit square at origin
C Note that the +0.5s were added when this code was
C included in DRIZZLE
      DO I = 1, 4
         PX(I) = X(I) - IS +0.5D0
         PY(I) = Y(I) - JS +0.5D0
      ENDDO

C For each line in the polygon (or at this stage, input quadrilateral)
C calculate the area common to the unit square (allow negative area for
C subsequent `vector' addition of subareas).
      SUM = 0.0D0
      DO I = 1, 3
         SUM = SUM + SGAREA (PX(I), PY(I), PX(I+1), PY(I+1), IS, JS)
      ENDDO

      SUM = SUM + SGAREA (PX(4), PY(4), PX(1), PY(1), IS, JS)
      DAREA = SUM

      RETURN
      END

      DOUBLE PRECISION FUNCTION SGAREA (X1, Y1, X2, Y2, IS, JS)
C
C To calculate area under a line segment within unit square at origin.
C This is used by BOXER
C
      IMPLICIT NONE
      DOUBLE PRECISION X1, Y1, X2, Y2

      INTEGER IS,JS
      DOUBLE PRECISION M, C, DX
      DOUBLE PRECISION XLO, XHI, YLO, YHI, XTOP
      LOGICAL NEGDX

      DX = X2 - X1

C Trap vertical line
      IF (DX .EQ. 0.0D0) THEN
         SGAREA = 0.0D0
         GO TO 80
      ENDIF

C Order the two input points in x
      IF (X1 .LT. X2) THEN
         XLO = X1
         XHI = X2
      ELSE
         XLO = X2
         XHI = X1
      ENDIF

C And determine the bounds ignoring y for now
      IF (XLO .GE. 1.0D0) THEN
         SGAREA = 0.0D0
         GO TO 80
      ENDIF

      IF (XHI .LE. 0.0D0) THEN
         SGAREA = 0.0D0
         GO TO 80
      ENDIF

      XLO = MAX (XLO, 0.0D0)
      XHI = MIN (XHI, 1.0D0)

C Now look at y
C basic info about the line y = mx + c
      NEGDX = (DX .LT. 0.0D0)
      M     = (Y2 - Y1) / DX
      C     = Y1 - M * X1
      YLO = M * XLO + C
      YHI = M * XHI + C

C Trap segment entirely below axis
      IF (YLO .LE. 0.0D0 .AND. YHI .LE. 0.0D0) THEN
         SGAREA = 0.0D0
         GO TO 80
      ENDIF

C Adjust bounds if segment crosses axis (to exclude anything below axis)
      IF (YLO .LT. 0.0D0) THEN
         YLO = 0.0D0
         XLO = -C/M
      ENDIF
      IF (YHI .LT. 0.0D0) THEN
         YHI = 0.0D0
         XHI = -C/M
      ENDIF

C There are four possibilities: both y below 1, both y above 1
C and one of each.

      IF (YLO .GE. 1.0D0 .AND. YHI .GE. 1.0D0) THEN

C Line segment is entirely above square
         IF (NEGDX) THEN
            SGAREA = XLO - XHI
         ELSE
            SGAREA = XHI - XLO
         ENDIF
         GO TO 80
      ENDIF

      IF (YLO .LE. 1.0D0 .AND. YHI .LE. 1.0D0) THEN

C Segment is entirely within square
         IF (NEGDX) THEN
            SGAREA = 0.5D0 * (XLO-XHI) * (YHI+YLO)
         ELSE
            SGAREA = 0.5D0 * (XHI-XLO) * (YHI+YLO)
         END IF
         GO TO 80
      ENDIF

C otherwise it must cross the top of the square
      XTOP = (1.0D0 - C) / M

      IF (YLO .LT. 1.0D0) THEN
         IF (NEGDX) THEN
            SGAREA = -(0.5D0 * (XTOP-XLO) * (1.0D0+YLO) + XHI - XTOP)
         ELSE
            SGAREA = 0.5D0 * (XTOP-XLO) * (1.0D0+YLO) + XHI - XTOP
         END IF
         GO TO 80
      ENDIF

      IF (NEGDX) THEN
         SGAREA = -(0.5D0 * (XHI-XTOP) * (1.0D0+YHI) + XTOP-XLO)
      ELSE
         SGAREA = 0.5D0 * (XHI-XTOP) * (1.0D0+YHI) + XTOP-XLO
      END IF

   80 CONTINUE

      RETURN
      END