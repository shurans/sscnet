classdef quaternion
% classdef quaternion, implements quaternion mathematics and 3D rotations
%
% Properties (SetAccess = protected):
%  e(4,1)   components, basis [1; i; j; k]: e(1) + i*e(2) + j*e(3) + k*e(4)
%           i*j=k, j*i=-k, j*k=i, k*j=-i, k*i=j, i*k=-j, i*i = j*j = k*k = -1
%
% Constructors:
%  q  = quaternion              scalar zero quaternion, q.e = [0;0;0;0]
%  q  = quaternion(x)           x is a matrix size [4,s1,s2,...] or [s1,4,s2,...],
%                               q is size [s1,s2,...], q(i1,i2,...).e = ...
%                               x(1:4,i1,i2,...) or x(i1,1:4,i2,...).'
%  q  = quaternion(v)           v is a matrix size [3,s1,s2,...] or [s1,3,s2,...],
%                               q is size [s1,s2,...], q(i1,i2,...).e = ...
%                               [0;v(1:3,i1,i2,...)] or [0;v(i1,1:3,i2,...).']
%  q  = quaternion(c)           c is a complex matrix size [s1,s2,...],
%                               q is size [s1,s2,...], q(i1,i2,...).e = ...
%                               [real(c(i1,i2,...));imag(c(i1,i2,...));0;0]
%  q  = quaternion(x1,x2)       x1,x2 are matrices size [s1,s2,...] or scalars,
%                               q(i1,i2,...).e = [x1(i1,i2,...);x2(i1,i2,...);0;0]
%  q  = quaternion(v1,v2,v3)    v1,v2,v3 matrices size [s1,s2,...] or scalars,
%                               q(i1,i2,...).e = [0;v1(i1,i2,...);v2(i1,i2,...);...
%                               v3(i1,i2,...)]
%  q  = quaternion(x1,x2,x3,x4) x1,x2,x3,x4 matrices size [s1,s2,...] or scalars,
%                               q(i1,i2,...).e = [x1(i1,i2,...);x2(i1,i2,...);...
%                               x3(i1,i2,...);x4(i1,i2,...)]
%
% Quaternion array constructor methods:
%  q  = quaternion.eye(N)       quaternion NxN identity matrix
%  q  = quaternion.nan(siz)     q(:).e = [NaN;NaN;NaN;NaN]
%  q  = quaternion.ones(siz)    q(:).e = [1;0;0;0]
%  q  = quaternion.rand(siz)    uniform random quaternions, NOT normalized
%                               to 1, 0 <= q.e(1) <= 1, -1 <= q.e(2:4) <= 1
%  q  = quaternion.randRot(siz) random quaternions uniform in rotation space
%  q  = quaternion.zeros(siz)   q(:).e = [0;0;0;0]
%
% Rotation constructor methods (all lower case):
%  q  = quaternion.angleaxis(angle,axis)
%                               angle is an array in radians, axis is an array
%                               of vectors size [3,s1,s2,...] or [s1,3,s2,...],
%                               q is size [s1,s2,...], quaternions normalized to 1
%                               equivalent to rotations about axis by angle
%  q  = quaternion.eulerangles(axes,angles) or
%  q  = quaternion.eulerangles(axes,ang1,ang2,ang3)
%                               axes is a string array or cell string array,
%                               '123' = 'xyz' = 'XYZ' = 'ijk', etc.,
%                               angles is an array of Euler angles in radians,
%                               size [3,s1,s2,...] or [s1,3,s2,...], or
%                               (ang1, ang2, ang3) are arrays or scalars of
%                               Euler angles in radians, q is size
%                               [s1,s2,...], quaternions normalized to 1
%                               equivalent to Euler Angle rotations
%  q  = quaternion.rotateutov(u,v,dimu,dimv)
%                               quaternions normalized to 1 that rotate 3
%                               element vectors u into the directions of 3
%                               element vectors v
%  q  = quaternion.rotationmatrix(R)
%                               R is an array of rotation or Direction Cosine
%                               Matrices size [3,3,s1,s2,...] with det(R) == 1,
%                               q(i1,i2,...) = quaternions normalized to 1,
%                               equivalent to R(1:3,1:3,i1,i2,...)
%
% Rotation methods (Mixed Case):
%  [angle,axis] = AngleAxis(q)  angles in radians, unit vector rotation axes
%                               equivalent to q
%  qd = Derivative(q,w)         quaternion derivatives, w are 3 component
%                               angular velocity vectors, qd = 0.5*q*quaternion(w)
%  angles = EulerAngles(q,axes) angles are 3 Euler angles equivalent to q, axes
%                               are strings or cell strings, '123' = 'xyz', etc.
%  [omega,axis] = OmegaAxis(q,t,dim)
%                               instantaneous angular velocities and rotation axes
%  PlotRotation(q,interval)     plot columns of rotation matrices of q,
%                               pause interval between figure updates in seconds
%  [q1,w1,t1] = PropagateEulerEq(q0,w0,I,t,@torque,odeoptions)
%                               Euler equation numerical propagator, see
%                               help quaternion.PropagateEulerEq
%  vp = RotateVector(q,v,dim)   vp are 3 component vectors, rotations q acting
%                               on vectors v, uses rotation matrix multiplication
%  vp = RotateVectorQ(q,v,dim)  vp are 3 component vectors, rotations q acting
%                               on vectors v, uses quaternion multiplication,
%                               RotateVector is 7 times faster than RotateVectorQ
%  R  = RotationMatrix(q)       3x3 rotation matrices equivalent to q
%
% Note:
%  In all rotation operations, the rotations operate from left to right on
%  3x1 column vectors and create rotated vectors, not representations of
%  those vectors in rotated coordinate systems.
%  For Euler angles, '123' means rotate the vector about x first, about y
%  second, about z third, i.e.:
%  vp = rotate(z,angle(3)) * rotate(y,angle(2)) * rotate(x,angle(1)) * v
%
% Ordinary methods:
%  n  = abs(q)                  quaternion norm, n = sqrt( sum( q.e.^2 ))
%  q3 = bsxfun(func,q1,q2)      binary singleton expansion of operation func
%  c  = complex(q)              complex( real(q), imag(q) )
%  qc = conj(q)                 quaternion conjugate, qc.e =
%                               [q.e(1);-q.e(2);-q.e(3);-q.e(4)]
%  qt = ctranspose(q)           qt = q'; quaternion conjugate transpose,
%                               2-D (or scalar) q only
%  qp = cumprod(q,dim)          cumulative quaternion array product over
%                               dimension dim
%  qs = cumsum(q,dim)           cumulative quaternion array sum over dimension dim
%  qd = diff(q,ord,dim)         quaternion array difference, order ord, over
%                               dimension dim
%  ans = display(q)             'q = ( e(1) ) + i( e(2) ) + j( e(3) ) + k( e(4) )'
%  d  = dot(q1,q2)              quaternion element dot product, d = dot(q1.e,q2.e)
%  d  = double(q)               d = q.e; if size(q) == [s1,s2,...], size(d) ==
%                               [4,s1,s2,...]
%  l  = eq(q1,q2)               quaternion equality, l = all( q1.e == q2.e )
%  l  = equiv(q1,q2,tol)        quaternion rotational equivalence, within
%                               tolerance tol, l = (q1 == q2) | (q1 == -q2)
%  qe = exp(q)                  quaternion exponential, v = q.e(2:4), qe.e =
%                               exp(q.e(1))*[cos(|v|);v.*sin(|v|)./|v|]
%  ei = imag(q)                 imaginary e(2) components
%  qi = interp1(t,q,ti,method)  interpolate quaternion array
%  qi = inverse(q)              quaternion inverse, qi = conj(q)./norm(q).^2,
%                               q .* qi = qi .*.q = 1 for q ~= 0
%  l  = isequal(q1,q2,...)      true if equal sizes and values
%  l  = isequaln(q1,q2,...)     true if equal including NaNs
%  l  = isequalwithequalnans(q1,q2,...) true if equal including NaNs
%  l  = isfinite(q)             true if all( isfinite( q.e ))
%  l  = isinf(q)                true if any( isinf( q.e ))
%  l  = isnan(q)                true if any( isnan( q.e ))
%  ej = jmag(q)                 e(3) components
%  ek = kmag(q)                 e(4) components
%  q3 = ldivide(q1,q2)          quaternion left division, q3 = q1 \. q2 =
%                               inverse(q1) *. q2
%  ql = log(q)                  quaternion logarithm, v = q.e(2:4), ql.e =
%                               [log(|q|);v.*acos(q.e(1)./|q|)./|v|]
%  q3 = minus(q1,q2)            quaternion subtraction, q3 = q1 - q2
%  q3 = mldivide(q1,q2)         left division only defined for scalar q1
%  qp = mpower(q,p)             quaternion matrix power, qp = q^p, p scalar
%                               integer >= 0, q square quaternion matrix
%  q3 = mrdivide(q1,q2)         right division only defined for scalar q2
%  q3 = mtimes(q1,q2)           2-D matrix quaternion multiplication, q3 = q1 * q2
%  l  = ne(q1,q2)               quaternion inequality, l = ~all( q1.e == q2.e )
%  n  = norm(q)                 quaternion norm, n = sqrt( sum( q.e.^2 ))
%  [q,n] = normalize(q)         make quaternion norm == 1, unless q == 0,
%                               n = matrix of previous norms
%  q3 = plus(q1,q2)             quaternion addition, q3 = q1 + q2
%  qp = power(q,p)              quaternion power, qp = q.^p
%  qp = prod(q,dim)             quaternion array product over dimension dim
%  qp = product(q1,q2)          quaternion product of scalar quaternions,
%                               qp = q1 .* q2, noncommutative
%  q3 = rdivide(q1,q2)          quaternion right division, q3 = q1 ./ q2 =
%                               q1 .* inverse(q2)
%  er = real(q)                 real e(1) components
%  qs = slerp(q0,q1,t)          quaternion spherical linear interpolation
%  qr = sqrt(q)                 qr = q.^0.5, square root
%  qs = sum(q,dim)              quaternion array sum over dimension dim
%  q3 = times(q1,q2)            matrix component quaternion multiplication,
%                               q3 = q1 .* q2, noncommutative
%  qm = uminus(q)               quaternion negation, qm = -q
%  qp = uplus(q)                quaternion unitary plus, qp = +q
%  ev = vector(q)               vector e(2:4) components
%
% Author:
%  Mark Tincknell, MIT LL, 29 July 2011, revised 22 November 2013

properties (SetAccess = protected)
    e   = zeros(4,1);
end % properties

% Array constructors
methods
    function q = quaternion( varargin ) % (constructor)
        perm = [];
        sqz  = false;
        switch nargin

            case 0  % nargin == 0
                q.e = zeros(4,1);
                return;

            case 1  % nargin == 1
                siz = size( varargin{1} );
                nel = prod( siz );
                if nel == 0
                    q	= quaternion.empty;
                    return;
                elseif isa( varargin{1}, 'quaternion' )
                    q   = varargin{1};
                    return;
                elseif (nel == 1) || ~isreal( varargin{1}(:) )
                    for iel = nel : -1 : 1
                        q(iel).e = chop( [real(varargin{1}(iel)); ...
                                          imag(varargin{1}(iel)); ...
                                          0; ...
                                          0] );
                    end
                    q   = reshape( q, siz );
                    return;
                end
                [arg4, dim4, perm4] = finddim( varargin{1}, 4 );
                if dim4 > 0
                    siz(dim4)   = 1;
                    nel         = prod( siz );
                    if dim4 > 1
                        perm    = perm4;
                    else
                        sqz     = true;
                    end
                    for iel = nel : -1 : 1
                        q(iel).e = chop( arg4(:,iel) );
                    end
                else
                    [arg3, dim3, perm3] = finddim( varargin{1}, 3 );
                    if dim3 > 0
                        siz(dim3)   = 1;
                        nel         = prod( siz );
                        if dim3 > 1
                            perm    = perm3;
                        else
                            sqz     = true;
                        end
                        for iel = nel : -1 : 1
                            q(iel).e = chop( [0; arg3(:,iel)] );
                        end
                    else
                        error( 'Invalid input' );
                    end
                end

            case 2  % nargin == 2
% real-imaginary only (no j or k) inputs
                na  = cellfun( 'prodofsize', varargin );
                [nel, jel]  = max( na );
                if ~all( (na == 1) | (na == nel) )
                    error( 'All inputs must be singletons or have the same number of elements' );
                end
                siz = size( varargin{jel} );
                for iel = nel : -1 : 1
                    q(iel).e = chop( [varargin{1}(min(iel,na(1))); ...
                                      varargin{2}(min(iel,na(2))); ...
                                      0;
                                      0] );
                end

            case 3  % nargin == 3
% vector inputs (no real, only i, j, k)
                na  = cellfun( 'prodofsize', varargin );
                [nel, jel]  = max( na );
                if ~all( (na == 1) | (na == nel) )
                    error( 'All inputs must be singletons or have the same number of elements' );
                end
                siz = size( varargin{jel} );
                for iel = nel : -1 : 1
                    q(iel).e = chop( [0; ...
                                      varargin{1}(min(iel,na(1))); ...
                                      varargin{2}(min(iel,na(2))); ...
                                      varargin{3}(min(iel,na(3)))] );
                end

            otherwise   % nargin >= 4
                na  = cellfun( 'prodofsize', varargin );
                [nel, jel]  = max( na );
                if ~all( (na == 1) | (na == nel) )
                    error( 'All inputs must be singletons or have the same number of elements' );
                end
                siz = size( varargin{jel} );
                for iel = nel : -1 : 1
                    q(iel).e = chop( [varargin{1}(min(iel,na(1))); ...
                                      varargin{2}(min(iel,na(2))); ...
                                      varargin{3}(min(iel,na(3))); ...
                                      varargin{4}(min(iel,na(4)))] );
                end
        end % switch nargin

        if nel == 0
            q   = quaternion.empty;
        end
        q   = reshape( q, siz );
        if ~isempty( perm )
            q   = ipermute( q, perm );
        end
        if sqz
            q   = squeeze( q );
        end
    end % quaternion (constructor)

% Ordinary methods
    function n = abs( q )
        n   = q.norm;
    end % abs

    function q3 = bsxfun( func, q1, q2 )
% function q3 = bsxfun( func, q1, q2 )
% Binary Singleton Expansion for quaternion arrays. Apply the element by
% element binary operation specified by the function handle func to arrays
% q1 and q2. All dimensions of q1 and q2 must either agree or be length 1.
% Inputs:
%  func     function handle (e.g. @plus) of quaternion function or operator
%  q1(n1)   quaternion array
%  q2(n2)   quaternion array
% Output:
%  q3(n3)   quaternion array of function or operator outputs
%           size(q3) = max( size(q1), size(q2) )
        if ~isa( q1, 'quaternion' )
            q1  = quaternion( real(q1), imag(q1), 0, 0 );
        end
        if ~isa( q2, 'quaternion' )
            q2  = quaternion( real(q2), imag(q2), 0, 0 );
        end
        s1  = size( q1 );
        s2  = size( q2 );
        nd1 = length( s1 );
        nd2 = length( s2 );
        s1  = [s1, ones(1,nd2-nd1)];
        s2  = [s2, ones(1,nd1-nd2)];
        if ~all( (s1 == s2) | (s1 == 1) | (s2 == 1) )
            error( 'Non-singleton dimensions of q1 and q2 must match each other' );
        end
        c1  = num2cell( s1 );
        c2  = num2cell( s2 );
        s3  = max( s1, s2 );
        nd3 = length( s3 );
        n3  = prod( s3 );
        q3  = quaternion.nan( s3 );
        for i3 = 1 : n3
            [ix3{1:nd3}] = ind2sub( s3, i3 ); 
            ix1     = cellfun( @min, ix3, c1, 'UniformOutput', false );
            ix2     = cellfun( @min, ix3, c2, 'UniformOutput', false );
            q3(i3)  = func( q1(ix1{:}), q2(ix2{:}) );
        end
    end % bsxfun

    function c  = complex( q )
        c   = complex( real( q ), imag( q ));
    end % complex

    function qc = conj( q )
        d   = double( q );
        qc  = reshape( quaternion( d(1,:), -d(2,:), -d(3,:), -d(4,:) ), ...
                       size( q ));
    end % conj

    function qt = ctranspose( q )
        qt  = transpose( q.conj );
    end % ctranspose

    function qp = cumprod( q, dim )
% function qp = cumprod( q, dim )
% cumulative quaternion array product, dim defaults to first dimension of
% length > 1
        if isempty( q )
            qp  = q;
            return;
        end
        if (nargin < 2) || isempty( dim )
            [q, dim, perm]  = finddim( q, -2 );
        elseif dim > 1
            ndm  = ndims( q );
            perm = [ dim : ndm, 1 : dim-1 ];
            q    = permute( q, perm );
        end
        qp  = q;
        for is = 2 : size(q,1)
            qp(is,:) = qp(is-1,:) .* q(is,:);
        end
        if dim > 1
            qp  = ipermute( qp, perm );
        end
    end % cumprod

    function qs = cumsum( q, dim )
% function qs = cumsum( q, dim )
% cumulative quaternion array sum, dim defaults to first dimension of
% length > 1
        if isempty( q )
            qs  = q;
            return;
        end
        if (nargin < 2) || isempty( dim )
            [q, dim, perm]  = finddim( q, -2 );
        elseif dim > 1
            ndm  = ndims( q );
            perm = [ dim : ndm, 1 : dim-1 ];
            q    = permute( q, perm );
        end
        qs  = q;
        for is = 2 : size(q,1)
            qs(is,:) = qs(is-1,:) + q(is,:);
        end
        if dim > 1
            qs  = ipermute( qs, perm );
        end
    end % cumsum

    function qd = diff( q, ord, dim )
% function qd = diff( q, ord, dim )
% quaternion array difference, ord is the order of difference (default = 1)
% dim defaults to first dimension of length > 1
        if isempty( q )
            qd  = q;
            return;
        end
        if (nargin < 2) || isempty( ord )
            ord = 1;
        end
        if ord <= 0
            qd  = q;
            return;
        end
        if (nargin < 3) || isempty( dim )
            [q, dim, perm]  = finddim( q, -2 );
        elseif dim > 1
            ndm  = ndims( q );
            perm = [ dim : ndm, 1 : dim-1 ];
            q    = permute( q, perm );
        end
        siz = size( q );
        if siz(1) <= 1
            qd  = quaternion.empty;
            return;
        end
        qd  = quaternion.zeros( [(siz(1)-1), siz(2:end)] );
        for is = 1 : siz(1)-1
            qd(is,:) = q(is+1,:) - q(is,:);
        end
        ord = ord - 1;
        if ord > 0
            qd  = diff( qd, ord, 1 );
        end
        if dim > 1
            qd  = ipermute( qd, perm );
        end
    end % diff

    function display( q )
        if ~isequal( get(0,'FormatSpacing'), 'compact' )
            disp(' ');
        end
        if isempty( q )
            fprintf( '%s \t= ([]) + i([]) + j([]) + k([])\n', inputname(1) )
            return;
        end
        siz = size( q );
        nel = [1 cumprod( siz )];
        ndm = length( siz );
        for iel = 1 : nel(end)
            if nel(end) == 1
                sub = '';
            else
                sub = ')';
                jel = iel - 1;
                for idm = ndm : -1 : 1
                    idx = floor( jel / nel(idm) ) + 1;
                    sub = [',' int2str(idx) sub]; %#ok<AGROW>
                    jel = rem( jel, nel(idm) );
                end
                sub(1)  = '(';
            end
            fprintf( '%s%s \t= (%-12.5g) + i(%-12.5g) + j(%-12.5g) + k(%-12.5g)\n', ...
                inputname(1), sub, q(iel).e )
        end
    end % display

    function d = dot( q1, q2 )
% function d = dot( q1, q2 )
% quaternion element dot product: d = dot( q1.e, q2.e ), using binary
% singleton expansion of quaternion arrays
% dn = dot( q1, q2 )/( norm(q1) * norm(q2) ) is the cosine of the angle in
% 4D space between 4D vectors q1.e and q2.e
        d   = squeeze( sum( bsxfun( @times, double( q1 ), double( q2 )), 1 ));
    end % dot

    function d = double( q )
        siz = size( q );
        d   = reshape( [q.e], [4 siz] );
        d   = chop( d );
    end % double

    function l = eq( q1, q2 )
        if ~isa( q1, 'quaternion' )
            q1  = quaternion( real(q1), imag(q1), 0, 0 );
        end
        if ~isa( q2, 'quaternion' )
            q2  = quaternion( real(q2), imag(q2), 0, 0 );
        end
        si1 = size( q1 );
        si2 = size( q2 );
        ne1 = prod( si1 );
        ne2 = prod( si2 );
        if (ne1 == 0) || (ne2 == 0)
            l   = logical([]);
            return;
        elseif ne1 == 1
            siz = si2;
        elseif ne2 == 1
            siz = si1;
        elseif isequal( si1, si2 )
            siz = si1;
        else
            error( 'Matrix dimensions must agree' );
        end
        l   = bsxfun( @eq, [q1.e], [q2.e] );
        l   = reshape( all( l, 1 ), siz );
    end % eq

    function l = equiv( q1, q2, tol )
% function l  = equiv( q1, q2, tol )
% quaternion rotational equivalence, within tolerance tol,
% l = (q1 == q2) | (q1 == -q2)
% optional argument tol (default = eps) sets tolerance for difference
% from exact equality
        if ~isa( q1, 'quaternion' )
            q1  = quaternion( real(q1), imag(q1), 0, 0 );
        end
        if ~isa( q2, 'quaternion' )
            q2  = quaternion( real(q2), imag(q2), 0, 0 );
        end
        if (nargin < 3) || isempty( tol )
            tol = eps;
        end
        si1 = size( q1 );
        si2 = size( q2 );
        ne1 = prod( si1 );
        ne2 = prod( si2 );
        if (ne1 == 0) || (ne2 == 0)
            l   = logical([]);
            return;
        elseif ne1 == 1
            siz = si2;
        elseif ne2 == 1
            siz = si1;
        elseif isequal( si1, si2 )
            siz = si1;
        else
            error( 'Matrix dimensions must agree' );
        end
        dm  = chop( bsxfun( @minus, [q1.e], [q2.e] ), tol );
        dp  = chop( bsxfun( @plus,  [q1.e], [q2.e] ), tol );
        l   = all( (dm == 0) | (dp == 0), 1 );
        l   = reshape( l, siz );
    end % equiv

    function qe = exp( q )
% function qe = exp( q )
% quaternion exponential, v = q.e(2:4),
% qe.e = exp(q.e(1))*[cos(|v|);v.*sin(|v|)./|v|]
        d       = double( q );
        siz     = size( d );
        od      = ones( 1, ndims( q ));
        vn      = reshape( sqrt( sum( d(2:4,:).^2, 1 )), [1 siz(2:end)] );
        cv      = cos( vn );
        sv      = sin( vn );
        n0      = vn ~= 0;
        sv(n0)  = sv(n0) ./ vn(n0);
        sv      = repmat( sv, [3, od] );
        ex      = repmat( reshape( exp( d(1,:) ), [1 siz(2:end)] ), [4, od] );
        de      = ex .* [ cv; sv .* reshape( d(2:4,:), [3 siz(2:end)] )];
        qe      = reshape( quaternion( de(1,:), de(2,:), de(3,:), de(4,:) ), ...
                           size( q ));
    end % exp

    function ei = imag( q )
        siz = size( q );
        d   = double( q );
        ei  = reshape( d(2,:), siz );
    end % imag

    function qi = interp1( varargin )
% function qi = interp1( t, q, ti, method ) or
%          qi = q.interp1( t, ti, method )  or
%          qi = interp1( q, ti, method )
% Interpolate quaternion array. If q are rotation quaternions (i.e.
% normalized to 1), then -q is equivalent to q, and the sign of q to use as
% the second knot of the interpolation is chosen by which ever is closer to
% the first knot. Extrapolation (i.e. ti < min(t) or ti > max(t)) gives
% qi = quaternion.nan.
% Inputs:
%  t(nt)      array of ordinates (e.g. times); if t is not provided t=1:nt
%  q(nt,nq)   quaternion array
%  ti(ni)     array of query (interpolation) points, t(1) <= ti <= t(end)
%  method [OPTIONAL] 'slerp' or 'linear'; default = 'slerp'
% Output:
%  qi(ni,nq)  interpolated quaternion array
        nna = nnz( ~cellfun( @ischar, varargin ));
        im  = 4;
        if isa( varargin{1}, 'quaternion' )
            q   = varargin{1};
            siq = size( q );
            if nna == 2
                if isrow( q )
                    t = (1 : siq(2)).';
                else
                    t = (1 : siq(1)).';
                end
                ti  = varargin{2}(:);
                im  = 3;
            elseif isempty( varargin{2} )
                if isrow( q )
                    t = (1 : siq(2)).';
                else
                    t = (1 : siq(1)).';
                end
                ti  = varargin{3}(:);
            else
                t   = varargin{2}(:);
                ti  = varargin{3}(:);
            end
        elseif isa( varargin{2}, 'quaternion' )
            t   = varargin{1}(:);
            q   = varargin{2};
            ti  = varargin{3}(:);
            siq = size( q );
        else
            error( 'Input q must be a quaterion' );
        end
        neq = prod( siq );
        if neq == 0
            qi  = quaternion.empty;
            return;
        end
        nt  = numel( t );
        if siq(1) == nt
            dim = 1;
        else
            [q, dim, perm]  = finddim( q, nt );
            if dim == 0
                error( 'q must have a dimension the same size as t' );
            end
        end
        iNf = interp1( t, (1:nt).', ti );
        iN  = max( 1, min( nt-1, floor( iNf )));
        jN  = max( 2, min( nt,   ceil(  iNf )));
        iNm = repmat( iNf - iN, [1, neq / nt] );
% If q are rotation quaternions (i.e. all normalized to 1), then -q
% represents the same rotation. Pick the sign of +/-q that has the closest
% dot product to use as the second knot of the interpolation.
        qj  = q(jN,:);
        if all( abs( norm( q(:) ) - 1 ) <= eps(16) )
            qd  = dot( q(iN,:), qj );
            lq  = qd < -qd;
            qj(lq)  = -qj(lq);
        end
        if (length( varargin ) >= im) && ...
           (strncmpi( 'linear', varargin{im}, length( varargin{im} )))
            qi  = (1 - iNm) .* q(iN,:) + iNm .* qj;
        else
            qi  = slerp( q(iN,:), qj, iNm );
        end
        if length( siq ) > 2
            sin = siq;
            sin(dim) = numel( ti );
            sin = circshift( sin, [0, 1-dim] );
            qi  = reshape( qi, sin );
        end
        if dim > 1
            qi  = ipermute( qi, perm );
        end
    end % interp1

    function qi = inverse( q )
% function qi = inverse( q )
% quaternion inverse, qi = conj(q)/norm(q)^2, q*qi = qi*q = 1 for q ~= 0
        if isempty( q )
            qi  = q;
            return;
        end
        d   = double( q );
        d(2:4,:) = -d(2:4,:);
        n2  = repmat( sum( d.^2, 1 ), 4, ones( 1, ndims( d ) - 1 ));
        ne0 = n2 ~= 0;
        di  = Inf( size( d ));
        di(ne0)  = d(ne0) ./ n2(ne0);
        qi  = reshape( quaternion( di(1,:), di(2,:), di(3,:), di(4,:) ), ...
                       size( q ));
    end % inverse

    function l = isequal( q1, varargin )
% function l = isequal( q1, q2, ... )
        nar = numel( varargin );
        if nar == 0
            error( 'Not enough input arguments' );
        end
        l   = false;
        if ~isa( q1, 'quaternion' )
            q1  = quaternion( real(q1), imag(q1), 0, 0 );
        end
        si1 = size( q1 );
        for iar = 1 : nar
            si2 = size( varargin{iar} );
            if (length( si1 ) ~= length( si2 )) || ...
               ~all( si1 == si2 )
                return;
            else
                if ~isa( varargin{iar}, 'quaternion' )
                    q2  = quaternion( ...
                          real(varargin{iar}), imag(varargin{iar}), 0, 0 );
                else
                    q2  = varargin{iar};
                end
                if ~isequal( [q1.e], [q2.e] )
                    return;
                end
            end
        end
        l   = true;
    end % isequal

    function l = isequaln( q1, varargin )
% function l = isequaln( q1, q2, ... )
        nar = numel( varargin );
        if nar == 0
            error( 'Not enough input arguments' );
        end
        l   = false;
        if ~isa( q1, 'quaternion' )
            q1  = quaternion( real(q1), imag(q1), 0, 0 );
        end
        si1 = size( q1 );
        for iar = 1 : nar
            si2 = size( varargin{iar} );
            if (length( si1 ) ~= length( si2 )) || ...
               ~all( si1 == si2 )
                return;
            else
                if ~isa( varargin{iar}, 'quaternion' )
                    q2  = quaternion( ...
                          real(varargin{iar}), imag(varargin{iar}), 0, 0 );
                else
                    q2  = varargin{iar};
                end
                if ~isequaln( [q1.e], [q2.e] )
                    return;
                end
            end
        end
        l   = true;
    end % isequaln

    function l = isequalwithequalnans( q1, varargin )
% function l = isequalwithequalnans( q1, q2, ... )
        nar = numel( varargin );
        if nar == 0
            error( 'Not enough input arguments' );
        end
        l   = false;
        if ~isa( q1, 'quaternion' )
            q1  = quaternion( real(q1), imag(q1), 0, 0 );
        end
        si1 = size( q1 );
        for iar = 1 : nar
            si2 = size( varargin{iar} );
            if (length( si1 ) ~= length( si2 )) || ...
               ~all( si1 == si2 )
                return;
            else
                if ~isa( varargin{iar}, 'quaternion' )
                    q2  = quaternion( ...
                          real(varargin{iar}), imag(varargin{iar}), 0, 0 );
                else
                    q2  = varargin{iar};
                end
                if ~isequalwithequalnans( [q1.e], [q2.e] ) %#ok<FPARK>
                    return;
                end
            end
        end
        l   = true;
    end % isequalwithequalnans

    function l = isfinite( q )
% function l = isfinite( q ), l = all( isfinite( q.e ))
        d   = [q.e];
        l   = reshape( all( isfinite( d ), 1 ), size( q ));
    end % isfinite

    function l = isinf( q )
% function l = isinf( q ), l = any( isinf( q.e ))
        d   = [q.e];
        l   = reshape( any( isinf( d ), 1 ), size( q ));
    end % isinf

    function l = isnan( q )
% function l = isnan( q ), l = any( isnan( q.e ))
        d   = [q.e];
        l   = reshape( any( isnan( d ), 1 ), size( q ));
    end % isnan

    function ej = jmag( q )
        siz = size( q );
        d   = double( q );
        ej  = reshape( d(3,:), siz );
    end % jmag

    function ek = kmag( q )
        siz = size( q );
        d   = double( q );
        ek  = reshape( d(4,:), siz );
    end % kmag

    function q3 = ldivide( q1, q2 )
        if ~isa( q1, 'quaternion' )
            q1  = quaternion( real(q1), imag(q1), 0, 0 );
        end
        if ~isa( q2, 'quaternion' )
            q2  = quaternion( real(q2), imag(q2), 0, 0 );
        end
        si1 = size( q1 );
        si2 = size( q2 );
        ne1 = prod( si1 );
        ne2 = prod( si2 );
        if (ne1 == 0) || (ne2 == 0)
            q3  = quaternion.empty;
            return;
        elseif ~isequal( si1, si2 ) && (ne1 ~= 1) && (ne2 ~= 1)
            error( 'Matrix dimensions must agree' );
        end
        for iel = max( ne1, ne2 ) : -1 : 1
            q3(iel) = product( q1(min(iel,ne1)).inverse, ...
                               q2(min(iel,ne2)) );
        end
        if ne2 > ne1
            q3  = reshape( q3, si2 );
        else
            q3  = reshape( q3, si1 );
        end
    end % ldivide

    function ql = log( q )
% function ql = log( q )
% quaternion logarithm, v = q.e(2:4), ql.e = [log(|q|);v.*acos(q.e(1)./|q|)./|v|]
% logarithm of negative real quaternions is ql.e = [log(|q|);pi;0;0]
        d       = double( q );
        d2      = d.^2;
        siz     = size( d );
        od      = ones( 1, ndims( q ));
        [vn,qn] = deal( zeros( [1 siz(2:end)] ));
        vn(:)   = sqrt( sum( d2(2:4,:), 1 ));
        qn(:)   = sqrt( sum( d2(1:4,:), 1 ));
        lq      = log( qn );
        d1      = reshape( d(1,:), [1 siz(2:end)] );
        nq      = qn ~= 0;
        d1(nq)  = d1(nq) ./ qn(nq);
        ac      = acos( d1 );
        nv      = vn ~= 0;
        ac(nv)  = ac(nv) ./ vn(nv);
        ac      = reshape( repmat( ac, [3, od] ), 3, [] );
        va      = reshape( d(2:4,:) .* ac, [3 siz(2:end)] );
        nn      = (d1 < 0) & (vn == 0);
        va(1,nn)= pi;
        dl      = [ lq; va ];
        ql      = reshape( quaternion( dl(1,:), dl(2,:), dl(3,:), dl(4,:) ), ...
                           size( q ));
    end % log

    function q3 = minus( q1, q2 )
        if ~isa( q1, 'quaternion' )
            q1  = quaternion( real(q1), imag(q1), 0, 0 );
        end
        if ~isa( q2, 'quaternion' )
            q2  = quaternion( real(q2), imag(q2), 0, 0 );
        end
        si1 = size( q1 );
        si2 = size( q2 );
        ne1 = prod( si1 );
        ne2 = prod( si2 );
        if (ne1 == 0) || (ne2 == 0)
            q3  = quaternion.empty;
            return;
        elseif ne1 == 1
            siz = si2;
        elseif ne2 == 1
            siz = si1;
        elseif isequal( si1, si2 )
            siz = si1;
        else
            error( 'Matrix dimensions must agree' );
        end
        d3  = bsxfun( @minus, [q1.e], [q2.e] );
        q3  = quaternion( d3(1,:), d3(2,:), d3(3,:), d3(4,:) );
        q3  = reshape( q3, siz );
    end % minus

    function q3 = mldivide( q1, q2 )
% function q3 = mldivide( q1, q2 ), left division only defined for scalar q1
        if numel( q1 ) > 1
            error( 'Left matix division undefined for quaternion arrays' );
        end
        q3  = ldivide( q1, q2 );
    end % mldivide

    function qp = mpower( q, p )
% function qp = mpower( q, p ), quaternion matrix power
        siq = size( q );
        neq = prod( siq );
        nep = numel( p );
        if neq == 1
            qp  = power( q, p );
            return;
        elseif isa( p, 'quaternion' )
            error( 'Quaternion as matrix exponent is not defined' );
        end
        if (neq == 0) || (nep == 0)
            qp  = quaternion.empty;
            return;
        elseif (nep > 1) || (mod( p, 1 ) ~= 0) || (p < 0) || ...
               (numel( siq ) > 2) || (siq(1) ~= siq(2))
            error( 'Inputs must be a scalar non-negative integer power and a square quaternion matrix' );
        elseif p == 0
            qp  = quaternion.eye( siq(1) );
            return;
        end
        qp  = q;
        for ip = 2 : p
            qp  = qp * q;
        end
    end % mpower

    function q3 = mrdivide( q1, q2 )
% function q3 = mrdivide( q1, q2 ), right division only defined for scalar q2
        if numel( q2 ) > 1
            error( 'Right matix division undefined for quaternion arrays' );
        end
        q3  = rdivide( q1, q2 );
    end % mrdivide

    function q3 = mtimes( q1, q2 )
% function q3 = mtimes( q1, q2 )
% q3 = matrix quaternion product of 2-D conformable quaternion matrices q1
% and q2
        if ~isa( q1, 'quaternion' )
            q1  = quaternion( real(q1), imag(q1), 0, 0 );
        end
        if ~isa( q2, 'quaternion' )
            q2  = quaternion( real(q2), imag(q2), 0, 0 );
        end
        si1 = size( q1 );
        si2 = size( q2 );
        ne1 = prod( si1 );
        ne2 = prod( si2 );
        if (ne1 == 1) || (ne2 == 1)
            q3  = times( q1, q2 );
            return;
        end
        if (length( si1 ) ~= 2) || (length( si2 ) ~= 2)
            error( 'Input arguments must be 2-D' );
        end
        if si1(2) ~= si2(1)
            error( 'Inner matrix dimensions must agree' );
        end
        q3  = repmat( quaternion, [si1(1) si2(2)] );
        for i1 = 1 : si1(1)
            for i2 = 1 : si2(2)
                for i3 = 1 : si1(2)
                    q3(i1,i2) = q3(i1,i2) + product( q1(i1,i3), q2(i3,i2) );
                end
            end
        end
    end % mtimes

    function l = ne( q1, q2 )
        l   = ~eq( q1, q2 );
    end % ne

    function n = norm( q )
        n   = shiftdim( sqrt( sum( double( q ).^2, 1 )), 1 );
    end % norm

    function [q, n] = normalize( q )
% function [q, n] = normalize( q )
% q = quaternions with norm == 1 (unless q == 0), n = former norms
        siz = size( q );
        nel = prod( siz );
        if nel == 0
            if nargout > 1
                n   = zeros( siz );
            end
            return;
        elseif nel > 1
            nel = [];
        end
        d   = double( q );
        n   = sqrt( sum( d.^2, 1 ));
        if all( n(:) == 1 )
            if nargout > 1
                n   = shiftdim( n, 1 );
            end
            return;
        end
        n4  = repmat( n, 4, nel );
        ne0 = (n4 ~= 0) & (n4 ~= 1);
        d(ne0)  = d(ne0) ./ n4(ne0);
        q   = reshape( quaternion( d(1,:), d(2,:), d(3,:), d(4,:) ), siz );
        if nargout > 1
            n   = shiftdim( n, 1 );
        end
    end % normalize

    function q3 = plus( q1, q2 )
        if ~isa( q1, 'quaternion' )
            q1  = quaternion( real(q1), imag(q1), 0, 0 );
        end
        if ~isa( q2, 'quaternion' )
            q2  = quaternion( real(q2), imag(q2), 0, 0 );
        end
        si1 = size( q1 );
        si2 = size( q2 );
        ne1 = prod( si1 );
        ne2 = prod( si2 );
        if (ne1 == 0) || (ne2 == 0)
            q3  = quaternion.empty;
            return;
        elseif ne1 == 1
            siz = si2;
        elseif ne2 == 1
            siz = si1;
        elseif isequal( si1, si2 )
            siz = si1;
        else
            error( 'Matrix dimensions must agree' );
        end
        d3  = bsxfun( @plus, [q1.e], [q2.e] );
        q3  = quaternion( d3(1,:), d3(2,:), d3(3,:), d3(4,:) );
        q3  = reshape( q3, siz );
    end % plus

    function qp = power( q, p )
% function qp = power( q, p ), quaternion power
        siq = size( q );
        sip = size( p );
        neq = prod( siq );
        nep = prod( sip );
        if (neq == 0) || (nep == 0)
            qp  = quaternion.empty;
            return;
        elseif ~isequal( siq, sip ) && (neq ~= 1) && (nep ~= 1)
            error( 'Matrix dimensions must agree' );
        end
        qp  = exp( p .* log( q ));
    end % power

    function qp = prod( q, dim )
% function qp = prod( q, dim )
% quaternion array product over dimension dim
% dim defaults to first dimension of length > 1
        if isempty( q )
            qp  = q;
            return;
        end
        if (nargin < 2) || isempty( dim )
            [q, dim, perm]  = finddim( q, -2 );
        elseif dim > 1
            ndm  = ndims( q );
            perm = [ dim : ndm, 1 : dim-1 ];
            q    = permute( q, perm );
        end
        siz = size( q );
        qp  = reshape( q(1,:), [1 siz(2:end)] );
        for is = 2 : siz(1)
            qp(1,:) = qp(1,:) .* q(is,:);
        end
        if dim > 1
            qp  = ipermute( qp, perm );
        end
    end % prod

    function q3 = product( q1, q2 )
% function q3 = product( q1, q2 )
% q3 = quaternion product of scalar quaternions q1 and q2
        if ~isa( q1, 'quaternion' )
            q1  = quaternion( real(q1), imag(q1), 0, 0 );
        end
        if ~isa( q2, 'quaternion' )
            q2  = quaternion( real(q2), imag(q2), 0, 0 );
        end
        if (numel( q1 ) ~= 1) || (numel( q2 ) ~= 1)
            error( 'product not defined for arrays, use mtimes or times' );
        end
        ee  = q1.e * q2.e.';
        eo  = [ee(1,1) - ee(2,2) - ee(3,3) - ee(4,4); ...
               ee(1,2) + ee(2,1) + ee(3,4) - ee(4,3); ...
               ee(1,3) - ee(2,4) + ee(3,1) + ee(4,2); ...
               ee(1,4) + ee(2,3) - ee(3,2) + ee(4,1)];
        eo  = chop( eo );
        q3  = quaternion( eo(1), eo(2), eo(3), eo(4) );
    end % product

    function q3 = rdivide( q1, q2 )
        if ~isa( q1, 'quaternion' )
            q1  = quaternion( real(q1), imag(q1), 0, 0 );
        end
        if ~isa( q2, 'quaternion' )
            q2  = quaternion( real(q2), imag(q2), 0, 0 );
        end
        si1 = size( q1 );
        si2 = size( q2 );
        ne1 = prod( si1 );
        ne2 = prod( si2 );
        if (ne1 == 0) || (ne2 == 0)
            q3  = quaternion.empty;
            return;
        elseif ~isequal( si1, si2 ) && (ne1 ~= 1) && (ne2 ~= 1)
            error( 'Matrix dimensions must agree' );
        end
        for iel = max( ne1, ne2 ) : -1 : 1
            q3(iel) = product( q1(min(iel,ne1)), ...
                               q2(min(iel,ne2)).inverse );
        end
        if ne2 > ne1
            q3  = reshape( q3, si2 );
        else
            q3  = reshape( q3, si1 );
        end
    end % rdivide

    function er = real( q )
        siz = size( q );
        d   = double( q );
        er  = reshape( d(1,:), siz );
    end % real

    function qs = slerp( q0, q1, t )
% function qs = slerp( q0, q1, t )
% quaternion spherical linear interpolation, qs = q0.*(q0.inverse.*q1).^t,
% default t = 0.5; see http://en.wikipedia.org/wiki/Slerp
        if (nargin < 3) || isempty( t )
            t   = 0.5;
        end
        qs  = q0 .* (q0.inverse .* q1).^t;
    end % slerp

    function qr = sqrt( q )
        qr  = q.^0.5;
    end % sqrt

    function qs = sum( q, dim )
% function qs = sum( q, dim )
% quaternion array sum over dimension dim
% dim defaults to first dimension of length > 1
        if isempty( q )
            qs  = q;
            return;
        end
        if (nargin < 2) || isempty( dim )
            [q, dim, perm]  = finddim( q, -2 );
        elseif dim > 1
            ndm  = ndims( q );
            perm = [ dim : ndm, 1 : dim-1 ];
            q    = permute( q, perm );
        end
        siz = size( q );
        qs  = reshape( q(1,:), [1 siz(2:end)] );
        for is = 2 : siz(1)
            qs(1,:) = qs(1,:) + q(is,:);
        end
        if dim > 1
            qs  = ipermute( qs, perm );
        end
    end % sum

    function q3 = times( q1, q2 )
        if ~isa( q1, 'quaternion' )
            q1  = quaternion( real(q1), imag(q1), 0, 0 );
        end
        if ~isa( q2, 'quaternion' )
            q2  = quaternion( real(q2), imag(q2), 0, 0 );
        end
        si1 = size( q1 );
        si2 = size( q2 );
        ne1 = prod( si1 );
        ne2 = prod( si2 );
        if (ne1 == 0) || (ne2 == 0)
            q3  = quaternion.empty;
            return;
        elseif ~isequal( si1, si2 ) && (ne1 ~= 1) && (ne2 ~= 1)
            error( 'Matrix dimensions must agree' );
        end
        for iel = max( ne1, ne2 ) : -1 : 1
            q3(iel) = product( q1(min(iel,ne1)), q2(min(iel,ne2)) );
        end
        if ne2 > ne1
            q3  = reshape( q3, si2 );
        else
            q3  = reshape( q3, si1 );
        end
    end % times

    function qm = uminus( q )
        d   = -double( q );
        qm  = reshape( quaternion( d(1,:), d(2,:), d(3,:), d(4,:) ), ...
                       size( q ));
    end % uminus

    function qp = uplus( q )
        qp  = q;
    end % uplus

    function ev = vector( q )
        siz = size( q );
        d   = double( q );
        ev  = reshape( d(2:4,:), [3 siz] );
    end % vector

    function [angle, axis] = AngleAxis( q )
% function [angle, axis] = AngleAxis( q )  or  [angle, axis] = q.AngleAxis
% Construct angle-axis pairs equivalent to quaternion rotations
% Input:
%  q        quaternion array
% Outputs:
%  angle    rotation angles in radians, 0 <= angle <= 2*pi
%  axis     3xN or Nx3 rotation axis unit vectors
% Note: angle and axis are constructed so at least 2 out of 3 elements of
% axis are >= 0.
        siz         = size( q );
        ndm         = length( siz );
        [angle, s]  = deal( zeros( siz ));
        axis        = zeros( [3 siz] );
        nel         = prod( siz );
        if nel == 0
            return;
        end
        [q, n]      = normalize( q );
        d           = double( q );
        neg         = repmat( reshape( d(1,:) < 0, [1 siz] ), ...
                              [4, ones(1,ndm)] );
        d(neg)      = -d(neg);
        angle(1:end)= 2 * acos( d(1,:) );
        s(1:end)    = sin( 0.5 * angle );
        angle(n==0) = 0;
        s(s==0)     = 1;
        s3          = shiftdim( s, -1 );
        axis(1:end) = bsxfun( @rdivide, reshape( d(2:4,:), [3 siz] ), s3 );
        axis(1,(mod(angle,2*pi)==0)) = 1;
        angle       = chop( angle );
        axis        = chop( axis );
% Flip axis so at least 2 out of 3 elements are >= 0
        flip        = (sum( axis < 0, 1 ) > 1) | ...
                      ((sum( axis == 0, 1 ) == 2) & (any( axis < 0, 1 ) == 1));
        angle(flip) = 2 * pi - angle(flip);
        flip        = repmat( flip, [3, ones(1,ndm)] );
        axis(flip)  = -axis(flip);
        axis        = squeeze( axis );
    end % AngleAxis

    function qd = Derivative( varargin )
% function qd = Derivative( q, w )   or   qd = q.Derivative( w )
% Inputs:
%  q        quaternion array
%  w        3xN or Nx3 element angle rate vectors in radians/s
% Output:
%  qd       quaternion derivatives, qd = 0.5 * q * quaternion(w)
        if isa( varargin{1}, 'quaternion' )
            qd  = 0.5 .* varargin{1} .* quaternion( varargin{2} );
        else
            qd  = 0.5 .* varargin{2} .* quaternion( varargin{1} );
        end
    end % Derivative

    function angles = EulerAngles( varargin )
% function angles = EulerAngles( q, axes )   or   angles = q.EulerAngles( axes )
% Construct Euler angle triplets equivalent to quaternion rotations
% Inputs:
%  q        quaternion array
%  axes     axes designation strings (e.g. '123' = xyz) or cell strings
%           (e.g. {'123'})
% Output:
%  angles   3 element Euler Angle vectors in radians
        ics     = cellfun( @ischar, varargin );
        if any( ics )
            varargin{ics} = cellstr( varargin{ics} );
        else
            ics = cellfun( @iscellstr, varargin );
        end
        if ~any( ics )
            error( 'Must provide axes as a string (e.g. ''123'') or cell string (e.g. {''123''})' );
        end
        siv     = cellfun( @size, varargin, 'UniformOutput', false );
        axes    = varargin{ics};
        six     = siv{ics};
        nex     = prod( six );
        q       = varargin{~ics};
        siq     = siv{~ics};
        neq     = prod( siq );
        if neq == 1
            siz = six;
            nel = nex;
        elseif nex == 1
            siz = siq;
            nel = neq;
        elseif nex == neq
            siz = siq;
            nel = neq;
        else
            error( 'Must have compatible dimensions for quaternion and axes' );
        end
        angles  = zeros( [3 siz] );
        q       = normalize( q );
        for jel = 1 : nel
            iel = min( jel, neq );
            switch axes{min(jel,nex)}
                case {'121', 'xyx', 'XYX', 'iji'}
                    angles(1,iel) = atan2((q(iel).e(2).*q(iel).e(3)- ...
                        q(iel).e(4).*q(iel).e(1)),(q(iel).e(2).*q(iel).e(4)+ ...
                        q(iel).e(3).*q(iel).e(1)));
                    angles(2,iel) = acos(q(iel).e(1).^2+q(iel).e(2).^2- ...
                        q(iel).e(3).^2-q(iel).e(4).^2);
                    angles(3,iel) = atan2((q(iel).e(2).*q(iel).e(3)+ ...
                        q(iel).e(4).*q(iel).e(1)),(q(iel).e(3).*q(iel).e(1)- ...
                        q(iel).e(2).*q(iel).e(4)));
                case {'123', 'xyz', 'XYZ', 'ijk'}
                    angles(1,iel) = atan2(2.*(q(iel).e(2).*q(iel).e(1)+ ...
                        q(iel).e(4).*q(iel).e(3)),(q(iel).e(1).^2- ...
                        q(iel).e(2).^2-q(iel).e(3).^2+q(iel).e(4).^2));
                    angles(2,iel) = asin(2.*(q(iel).e(3).*q(iel).e(1)- ...
                        q(iel).e(2).*q(iel).e(4)));
                    angles(3,iel) = atan2(2.*(q(iel).e(2).*q(iel).e(3)+ ...
                        q(iel).e(4).*q(iel).e(1)),(q(iel).e(1).^2+ ...
                        q(iel).e(2).^2-q(iel).e(3).^2-q(iel).e(4).^2));
                case {'131', 'xzx', 'XZX', 'iki'}
                    angles(1,iel) = atan2((q(iel).e(2).*q(iel).e(4)+ ...
                        q(iel).e(3).*q(iel).e(1)),(q(iel).e(4).*q(iel).e(1)- ...
                        q(iel).e(2).*q(iel).e(3)));
                    angles(2,iel) = acos(q(iel).e(1).^2+q(iel).e(2).^2- ...
                        q(iel).e(3).^2-q(iel).e(4).^2);
                    angles(3,iel) = atan2((q(iel).e(2).*q(iel).e(4)- ...
                        q(iel).e(3).*q(iel).e(1)),(q(iel).e(2).*q(iel).e(3)+ ...
                        q(iel).e(4).*q(iel).e(1)));
                case {'132', 'xzy', 'XZY', 'ikj'}
                    angles(1,iel) = atan2(2.*(q(iel).e(2).*q(iel).e(1)- ...
                        q(iel).e(4).*q(iel).e(3)),(q(iel).e(1).^2- ...
                        q(iel).e(2).^2+q(iel).e(3).^2-q(iel).e(4).^2));
                    angles(2,iel) = asin(2.*(q(iel).e(2).*q(iel).e(3)+ ...
                        q(iel).e(4).*q(iel).e(1)));
                    angles(3,iel) = atan2(2.*(q(iel).e(3).*q(iel).e(1)- ...
                        q(iel).e(2).*q(iel).e(4)),(q(iel).e(1).^2+ ...
                        q(iel).e(2).^2-q(iel).e(3).^2-q(iel).e(4).^2));
                case {'212', 'yxy', 'YXY', 'jij'}
                    angles(1,iel) = atan2((q(iel).e(2).*q(iel).e(3)+ ...
                        q(iel).e(4).*q(iel).e(1)),(q(iel).e(2).*q(iel).e(1)- ...
                        q(iel).e(3).*q(iel).e(4)));
                    angles(2,iel) = acos(q(iel).e(1).^2-q(iel).e(2).^2+ ...
                        q(iel).e(3).^2-q(iel).e(4).^2);
                    angles(3,iel) = atan2((q(iel).e(2).*q(iel).e(3)- ...
                        q(iel).e(4).*q(iel).e(1)),(q(iel).e(2).*q(iel).e(1)+ ...
                        q(iel).e(3).*q(iel).e(4)));
                case {'213', 'yxz', 'YXZ', 'jik'}
                    angles(1,iel) = atan2(2.*(q(iel).e(3).*q(iel).e(1)- ...
                        q(iel).e(4).*q(iel).e(2)),(q(iel).e(1).^2- ...
                        q(iel).e(2).^2-q(iel).e(3).^2+q(iel).e(4).^2));
                    angles(2,iel) = asin(2.*(q(iel).e(2).*q(iel).e(1)+ ...
                        q(iel).e(3).*q(iel).e(4)));
                    angles(3,iel) = atan2(2.*(q(iel).e(4).*q(iel).e(1)- ...
                        q(iel).e(2).*q(iel).e(3)),(q(iel).e(1).^2- ...
                        q(iel).e(2).^2+q(iel).e(3).^2-q(iel).e(4).^2));
                case {'231', 'yzx', 'YZX', 'jki'}
                    angles(1,iel) = atan2(2.*(q(iel).e(2).*q(iel).e(4)+ ...
                        q(iel).e(3).*q(iel).e(1)),(q(iel).e(1).^2+ ...
                        q(iel).e(2).^2-q(iel).e(3).^2-q(iel).e(4).^2));
                    angles(2,iel) = asin(2.*(q(iel).e(4).*q(iel).e(1)- ...
                        q(iel).e(2).*q(iel).e(3)));
                    angles(3,iel) = atan2(2.*(q(iel).e(2).*q(iel).e(1)+ ...
                        q(iel).e(3).*q(iel).e(4)),(q(iel).e(1).^2- ...
                        q(iel).e(2).^2+q(iel).e(3).^2-q(iel).e(4).^2));
                case {'232', 'yzy', 'YZY', 'jkj'}
                    angles(1,iel) = atan2((q(iel).e(3).*q(iel).e(4)- ...
                        q(iel).e(2).*q(iel).e(1)),(q(iel).e(2).*q(iel).e(3)+ ...
                        q(iel).e(4).*q(iel).e(1)));
                    angles(2,iel) = acos(q(iel).e(1).^2-q(iel).e(2).^2+ ...
                        q(iel).e(3).^2-q(iel).e(4).^2);
                    angles(3,iel) = atan2((q(iel).e(2).*q(iel).e(1)+ ...
                        q(iel).e(3).*q(iel).e(4)),(q(iel).e(4).*q(iel).e(1)- ...
                        q(iel).e(2).*q(iel).e(3)));
                case {'312', 'zxy', 'ZXY', 'kij'}
                    angles(1,iel) = atan2(2.*(q(iel).e(2).*q(iel).e(3)+ ...
                        q(iel).e(4).*q(iel).e(1)),(q(iel).e(1).^2- ...
                        q(iel).e(2).^2+q(iel).e(3).^2-q(iel).e(4).^2));
                    angles(2,iel) = asin(2.*(q(iel).e(2).*q(iel).e(1)- ...
                        q(iel).e(3).*q(iel).e(4)));
                    angles(3,iel) = atan2(2.*(q(iel).e(2).*q(iel).e(4)+ ...
                        q(iel).e(3).*q(iel).e(1)),(q(iel).e(1).^2- ...
                        q(iel).e(2).^2-q(iel).e(3).^2+q(iel).e(4).^2));
                case {'313', 'zxz', 'ZXZ', 'kik'}
                    angles(1,iel) = atan2((q(iel).e(2).*q(iel).e(4)- ...
                        q(iel).e(3).*q(iel).e(1)),(q(iel).e(2).*q(iel).e(1)+ ...
                        q(iel).e(3).*q(iel).e(4)));
                    angles(2,iel) = acos(q(iel).e(1).^2-q(iel).e(2).^2- ...
                        q(iel).e(3).^2+q(iel).e(4).^2);
                    angles(3,iel) = atan2((q(iel).e(2).*q(iel).e(4)+ ...
                        q(iel).e(3).*q(iel).e(1)),(q(iel).e(2).*q(iel).e(1)- ...
                        q(iel).e(3).*q(iel).e(4)));
                case {'321', 'zyx', 'ZYX', 'kji'}
                    angles(1,iel) = atan2(2.*(q(iel).e(4).*q(iel).e(1)- ...
                        q(iel).e(2).*q(iel).e(3)),(q(iel).e(1).^2+ ...
                        q(iel).e(2).^2-q(iel).e(3).^2-q(iel).e(4).^2));
                    angles(2,iel) = asin(2.*(q(iel).e(2).*q(iel).e(4)+ ...
                        q(iel).e(3).*q(iel).e(1)));
                    angles(3,iel) = atan2(2.*(q(iel).e(2).*q(iel).e(1)- ...
                        q(iel).e(3).*q(iel).e(4)),(q(iel).e(1).^2- ...
                        q(iel).e(2).^2-q(iel).e(3).^2+q(iel).e(4).^2));
                case {'323', 'zyz', 'ZYZ', 'kjk'}
                    angles(1,iel) = atan2((q(iel).e(2).*q(iel).e(1)+ ...
                        q(iel).e(3).*q(iel).e(4)),(q(iel).e(3).*q(iel).e(1)- ...
                        q(iel).e(2).*q(iel).e(4)));
                    angles(2,iel) = acos(q(iel).e(1).^2-q(iel).e(2).^2- ...
                        q(iel).e(3).^2+q(iel).e(4).^2);
                    angles(3,iel) = atan2((q(iel).e(3).*q(iel).e(4)- ...
                        q(iel).e(2).*q(iel).e(1)),(q(iel).e(2).*q(iel).e(4)+ ...
                        q(iel).e(3).*q(iel).e(1)));
                otherwise
                    error( 'Invalid output Euler angle axes' );
            end % switch axes
        end % for iel
        angles  = chop( angles );
    end % EulerAngles

    function [omega, axis] = OmegaAxis( q, t, dim )
% function [omega, axis] = OmegaAxis( q, t, dim )  or
%          [omega, axis] = q.OmegaAxis( t, dim )
% Estimate instantaneous angular velocities and rotation axes from a time
% series of quaternions. The angular velocity vector omegav is computed by:
%  omegav(:,1) = vector( 2*log( q(1) * inverse(q(2)) )/(t(2) - t(1)) );
%  omegav(:,i) = vector(...
%     (log( q(i-1) * inverse(q(i)) ) + log( q(i) * inverse(q(i+1))) )/...
%     (0.5*(t(i+1) - t(i-1))) );
%  omegav(:,end) = vector( 2*log( q(end-1) * inverse(q(end)) )/...
%     (t(end) - t(end-1)) );
%  [axis, omega] = unitvector( omegav );
% Inputs:
%  q          array of normalized (rotation) quaternions
%  t   [OPT]  array of monotonically increasing (or decreasing) times.
%             if omitted or empty, unit time steps are assumed.
%             t must either be a vector with the same length as dimension
%             dim of q, or the same size as q.
%  dim [OPT]  dimension of q that is varying in time; if omitted or empty,
%             the first non-singleton dimension is used.
% Outputs:
%  omega      array of instantaneous angular velocities, radians/(unit time)
%             omega >= 0
%  axis       instantaneous 3D rotation axis unit vectors at each time
        if isempty( q )
            omega   = [];
            axis    = [];
            return;
        end
        if (nargin < 3) || isempty( dim )
            if (nargin > 1) && ~isempty( t )
                siq = size( q );
                sit = size( t );
                if isequal( siq, sit )
                    dim   = find( siq > 1, 1 );
                else
                    dim   = find( siq == length( t ), 1 );
                end
                if isempty( dim )
                    error( 'size of t must agree with at least one dimension of q' );
                elseif dim > 1
                    ndm   = ndims( q );
                    perm  = [ dim : ndm, 1 : dim-1 ];
                    q     = permute( q, perm );
                    if isequal( siq, sit )
                        t = permute( t, perm );
                    end
                end
            else
                [q, dim, perm] = finddim( q, -2 );
                if dim == 0
                    omega = 0;
                    axis  = unitvector( q.e(2:4), 1 );
                    return;
                end
            end
        elseif dim > 1
            ndm   = ndims( q );
            perm  = [ dim : ndm, 1 : dim-1 ];
            q     = permute( q, perm );
        end
        n   = norm( q );
        if ~all( abs( n(:) - 1 ) < eps(16) )
            error( 'q must be normalized' );
        end
        siq = size( q );
        if (nargin < 2) || isempty( t )
            t   = repmat( (0 : (siq(1)-1)).', [1 siq(2:end)] );
        elseif length( t ) == siq(1)
            t   = repmat( t(:), [1 siq(2:end)] );
        elseif ~isequal( siq, size( t ))
            error( 'size of t must match size of q' );
        end
        dt            = zeros( siq );
        difft         = diff( t, 1 );
        dt(1,:)       = difft(1,:);
        dt(2:end-1,:) = 0.5 *( difft(1:end-1,:) + difft(2:end,:) );
        dt(end,:)     = difft(end,:);
        dq            = quaternion.zeros( siq );
        q1iq2         = q(1:end-1,:) .* inverse( q(2:end,:) );
        neg           = real( q1iq2 ) < 0;
        q1iq2(neg)    = -q1iq2(neg);    % keep real element >= 0
        derivq        = log( q1iq2 );
        dq(1,:)       = 2 .* derivq(1,:);
        dq(2:end-1,:) = derivq(1:end-1,:) + derivq(2:end,:);
        dq(end,:)     = 2 .* derivq(end,:);
        omegav        = vector( dq );   % angular velocity vectors
        [axis, omega] = unitvector( omegav, 1 );
        omega         = reshape( omega(1,:), siq )./ dt;
        axis          = -axis;
        if dim > 1
            axis      = ipermute( axis, [1, 1+perm] );
            omega     = ipermute( omega, perm );
        end
    end % OmegaAxis

    function PlotRotation( q, interval )
% function PlotRotation( q, interval )   or   q.PlotRotation( interval )
% Inputs:
%  q          quaternion array
%  interval   pause between figure updates in seconds, default = 0.1
% Output:
%  figure plotting the 3 Cartesian axes orientations for the series of
%  quaternions in array q
        if (nargin < 2) || isempty( interval )
            interval    = 0.1;
        end
        nel = numel( q );
        or  = zeros(1,3);
        ax  = eye(3);
        alx = zeros( nel, 3, 3 );
        figure;
        for iel = 1 : nel
%           plot3( [ or; ax(:,1).' ], [ or ; ax(:,2).' ], [ or; ax(:,3).' ], ':' );
            plot3( [ or; ax(1,:) ], [ or ; ax(2,:) ], [ or; ax(3,:) ], ':' );
            hold on
            set( gca, 'Xlim', [-1 1], 'Ylim', [-1 1], 'Zlim', [-1 1] );
            xlabel( 'x' );
            ylabel( 'y' );
            zlabel( 'z' );
            grid on
            nax = q(iel).RotationMatrix;
            alx(iel,:,:)    = nax; 
%           plot3( [ or; nax(:,1).' ], [ or ; nax(:,2).' ], [ or; nax(:,3).' ], '-', 'LineWidth', 2 );
            plot3( [ or; nax(1,:) ], [ or ; nax(2,:) ], [ or; nax(3,:) ], '-', 'LineWidth', 2 );
%           plot3( alx(1:iel,:,1), alx(1:iel,:,2), alx(1:iel,:,3), '*' );
            plot3( squeeze(alx(1:iel,1,:)), squeeze(alx(1:iel,2,:)), squeeze(alx(1:iel,3,:)), '*' );
            if interval
                pause( interval );
            end
            hold off
        end
    end % PlotRotation

    function [q1, w1, t1] = PropagateEulerEq( q0, w0, I, t, torque, varargin )
% function [q1, w1, t1] = PropagateEulerEq( q0, w0, I, t, torque, odeoptions )
% Inputs:
%  q0           initial orientation quaternion (normalized, scalar)
%  w0(3)        initial body frame angular velocity vector
%  I(3)         principal body moments of inertia (if no torque, only
%               ratios of elements of I are used)
%  t(nt)        initial and subsequent (or previous) times t = [t0,t1,...]
%               (monotonic)
%  @torque [OPTIONAL] function handle to calculate torque vector:
%               tau(1:3) = torque( t, y ), where y = [q.e(1:4); w(1:3)]
%  odeoptions [OPTIONAL] ode45 options
% Outputs:
%  q1(1,nt)     array of normalized quaternions at times t1
%  w1(3,nt)     array of body frame angular velocity vectors at times t1
%  t1(1,nt)     array of output times
% Calls:
%  Derivative   quaternion derivative method
%  odeset       matlab ode options setter
%  ode45        matlab ode numerical differential equation integrator
%  torque [OPTIONAL] user-supplied torque as function of time, orientation,
%               and angular rates; default is no torque
% Author:
%  Mark Tincknell, 20 December 2010
%         modified 25 July 2012, enforce normalization of q0 and q1
        options = odeset( varargin{:} );
        q0      = q0.normalize;
        y0      = [q0.e; w0(:)];
        I0      = [ (I(2) - I(3)) / I(1);
                    (I(3) - I(1)) / I(2);
                    (I(1) - I(2)) / I(3) ];
        [T, Y]  = ode45( @Euler, t, y0, options );
            function yd = Euler( ti, yi )
                qi  = quaternion( yi(1), yi(2), yi(3), yi(4) );
                wi  = yi(5:7);
                qd  = double( qi.Derivative( wi ));
                wd  = [ wi(2) * wi(3) * I0(1);
                        wi(3) * wi(1) * I0(2);
                        wi(1) * wi(2) * I0(3) ];
                if exist( 'torque', 'var' ) && isa( torque, 'function_handle' )
                    tau = torque( ti, yi );
                    wd  = tau(:) ./ I + wd;
                end
                yd  = [ qd; wd ];
            end
        if numel(t) == 2
            nT  = 2;
            T   = [T(1);   T(end)];
            Y   = [Y(1,:); Y(end,:)];
        else
            nT  = length(T);
        end
        q1      = repmat( quaternion, [1 nT] );
        w1      = zeros( [3 nT] );
        t1      = T(:).';
        for it = 1 : nT
            q1(it)   = quaternion( Y(it,1), Y(it,2), Y(it,3), Y(it,4) );
            w1(:,it) = Y(it,5:7).';
        end
        q1      = q1.normalize;
        neg     = real( q1 ) < 0;
        q1(neg) = -q1(neg);     % keep real element >= 0
    end % PropagateEulerEq

    function vp = RotateVector( varargin )
% function vp = RotateVector( q, v, dim ) or
%          vp = q.RotateVector( v, dim )
% 3x3 rotation matrices are created from q and matrix multiplication
% rotates v into vp. RotateVector is 7 times faster than RotateVectorQ.
% Inputs:
%  q        quaternion array
%  v        3xN or Nx3 element Cartesian vectors
%  dim [OPTIONAL] dimension of v with size 3 to rotate
% Output:
%  vp       3xN or Nx3 element rotated vectors
        if nargin < 2
            error( 'RotateVector method requires 2 inputs: a vector and a quaternion' );
        end
        if isa( varargin{1}, 'quaternion' )
            q   = varargin{1};
            v   = varargin{2};
        else
            v   = varargin{1};
            q   = varargin{2};
        end
        if (nargin > 2) && ~isempty( varargin{3} )
            dim = varargin{3};
            if size( v, dim ) ~= 3
                error( 'Dimension dim of vector v must be size 3' );
            end
            if dim > 1
                ndm  = ndims( v );
                perm = [ dim : ndm, 1 : dim-1 ];
                v    = permute( v, perm );
            end
        else
            [v, dim, perm] = finddim( v, 3 );
            if dim == 0
                error( 'v must have a dimension of size 3' );
            end
        end
        sip = size( v );
        v   = reshape( v, 3, [] );
        nev = prod( sip )/ 3;
        R   = q.RotationMatrix;
        siq = size( q );
        neq = prod( siq );
        if neq == nev
            vp  = zeros( sip );
            for iel = 1 : neq
                vp(:,iel)   = R(:,:,iel) * v(:,iel);
            end
            if dim > 1
                vp  = ipermute( vp, perm );
            end
        elseif nev == 1
            siz = [3 siq];
            vp  = zeros( siz );
            for iel = 1 : neq
                vp(:,iel)   = R(:,:,iel) * v;
            end
            if siz(2) == 1
                vp  = squeeze( vp );
            end
        elseif neq == 1
            vp  = R * v;
            vp  = reshape( vp, sip );
            if dim > 1
                vp  = ipermute( vp, perm );
            end
        else
            error( 'q and v must have compatible dimensions' );
        end
    end % RotateVector

    function vp = RotateVectorQ( varargin )
% function vp = RotateVectorQ( q, v, dim ) or
%          vp = q.RotateVectorQ( v, dim )
% quaternions are created from v and quaternion multiplication rotates v
% into vp. RotateVector is 7 times faster than RotateVectorQ.
% Inputs:
%  q        quaternion array
%  v        3xN or Nx3 element Cartesian vectors
%  dim [OPTIONAL] dimension of v with size 3 to rotate
% Output:
%  vp       3xN or Nx3 element rotated vectors
        if nargin < 2
            error( 'RotateVectorQ method requires 2 inputs: a vector and a quaternion' );
        end
        if isa( varargin{1}, 'quaternion' )
            q   = varargin{1};
            v   = varargin{2};
        else
            v   = varargin{1};
            q   = varargin{2};
        end
        siv = size( v );
        if (nargin > 2) && ~isempty( varargin{3} )
            dim = varargin{3};
            if size( v, dim ) ~= 3
                error( 'Dimension dim of vector v must be size 3' );
            end
            if dim > 1
                ndm  = ndims( v );
                perm = [ dim : ndm, 1 : dim-1 ];
                v    = permute( v, perm );
            end
        else
            [v, dim, perm] = finddim( v, 3 );
            if dim == 0
                error( 'v must have a dimension of size 3' );
            end
        end
        sip = size( v );
        qv  = quaternion( v(1,:), v(2,:), v(3,:) );
        qv  = reshape( qv, [1 sip(2:end)] );
        if dim > 1
            qv  = ipermute( qv, perm );
        end
        q   = q.normalize;
        qp  = q .* qv .* q.conj;
        dp  = qp.double;
        nev = prod( siv )/ 3;
        sqz = false;
        if nev == 1
            siz = [3 size(q)];
            if siz(2) == 1
                sqz = true;
            end
        else
            siz = siv;
        end
        vp  = reshape( dp(2:4,:), siz );
        if sqz
            vp  = squeeze( vp );
        end
    end % RotateVectorQ

    function R = RotationMatrix( q )
% function R = RotationMatrix( q )   or   R = q.RotationMatrix
% Construct rotation (or direction cosine) matrices from quaternions
% Input:
%  q        quaternion array
% Output:
%  R        3x3xN rotation (or direction cosine) matrices
        siz = size( q );
        R   = zeros( [3 3 siz] );
        nel = prod( siz );
        q   = normalize( q );
        for iel = 1 : nel
            e11 = q(iel).e(1)^2;
            e12 = q(iel).e(1) * q(iel).e(2);
            e13 = q(iel).e(1) * q(iel).e(3);
            e14 = q(iel).e(1) * q(iel).e(4);
            e22 = q(iel).e(2)^2;
            e23 = q(iel).e(2) * q(iel).e(3);
            e24 = q(iel).e(2) * q(iel).e(4);
            e33 = q(iel).e(3)^2;
            e34 = q(iel).e(3) * q(iel).e(4);
            e44 = q(iel).e(4)^2;
            R(:,:,iel)  = ...
             [ e11 + e22 - e33 - e44, 2*(e23 - e14), 2*(e24 + e13); ...
               2*(e23 + e14), e11 - e22 + e33 - e44, 2*(e34 - e12); ...
               2*(e24 - e13), 2*(e34 + e12), e11 - e22 - e33 + e44 ];
        end
        R   = chop( R );
    end % RotationMatrix
end % methods

% Static methods
methods(Static)
    function q = angleaxis( angle, axis )
% function q = quaternion.angleaxis( angle, axis )
% Construct quaternions from rotation axes and rotation angles
% Inputs:
%  angle    array of rotation angles in radians
%  axis     3xN or Nx3 array of axes (need not be unit vectors)
% Output:
%  q        quaternion array
        sig = size( angle );
        six = size( axis );
        [axis, dim, perm] = finddim( axis, 3 );
        if dim == 0
            error( 'axis must have a dimension of size 3' );
        end
        neg = prod( sig );
        nex = prod( six )/ 3;
        if neg == 1
            siz     = six;
            siz(dim)= 1;
            nel     = nex;
        elseif nex == 1
            siz     = sig;
            nel     = neg;
        elseif nex == neg
            siz     = sig;
            nel     = neg;
        else
            error( 'angle and axis must have compatible sizes' );
        end
        for iel = nel : -1 : 1
            d(:,iel) = AngAxis2e( angle(min(iel,neg)), axis(:,min(iel,nex)) );
        end
        q   = quaternion( d(1,:), d(2,:), d(3,:), d(4,:) );
        q   = reshape( q, siz );
        if neg == 1
            q   = ipermute( q, perm );
        end
    end % quaternion.angleaxis

    function q = eulerangles( varargin )
% function q = quaternion.eulerangles( axes, angles )  OR
% function q = quaternion.eulerangles( axes, ang1, ang2, ang3 )
% Construct quaternions from triplets of axes and Euler angles
% Inputs:
%  axes                 string array or cell string array
%                       '123' = 'xyz' = 'XYZ' = 'ijk', etc.
%  angles               3xN or Nx3 array of angles in radians  OR
%  ang1, ang2, ang3     arrays of angles in radians
% Output:
%  q                    quaternion array
        ics = cellfun( @ischar, varargin );
        if any( ics )
            varargin{ics} = cellstr( varargin{ics} );
        else
            ics = cellfun( @iscellstr, varargin );
        end
        siv     = cellfun( @size, varargin, 'UniformOutput', false );
        axes    = varargin{ics};
        six     = siv{ics};
        nex     = prod( six );
        dim     = 1;

        if nargin == 2  % angles is 3xN or Nx3 array
            angles  = varargin{~ics};
            sig     = siv{~ics};
            [angles, dim, perm] = finddim( angles, 3 );
            if dim == 0
                error( 'Must supply 3 Euler angles' );
            end
            sig(dim)    = 1;
            neg         = prod( sig );
            if nex == 1
                siz     = sig;
            elseif neg == 1
                siz     = six;
            elseif nex == neg
                siz     = sig;
            end
            nel = prod( siz );
            for iel = nel : -1 : 1
                q(iel)  = EulerAng2q( axes{min(iel,nex)}, ...
                                      angles(:,min(iel,neg)) );
            end

        elseif nargin == 4  % each of 3 angles is separate input argument
            angles  = varargin(~ics);
            na      = cellfun( 'prodofsize', angles );
            [neg, jeg] = max( na );
            if ~all( (na == 1) | (na == neg) )
                error( 'All angles must be singletons or have the same number of elements' );
            end
            sig = size( angles{jeg} );
            if nex == 1
                siz     = sig;
            elseif neg == 1
                siz     = six;
            elseif nex == neg
                siz     = sig;
            end
            nel = prod( siz );
            for iel = nel : -1 : 1
                q(iel)  = EulerAng2q( axes{min(iel,nex)}, ...
                                      [angles{1}(min(iel,na(1))), ...
                                       angles{2}(min(iel,na(2))), ...
                                       angles{3}(min(iel,na(3)))] );
            end
        else
            error( 'Must supply either 2 or 4 input arguments' );
        end % if nargin

        q   = reshape( q, siz );
        if (dim > 1) && isequal( siz, sig )
            q   = ipermute( q, perm );
        end
        if ~ismatrix( q ) && (size( q, 1 ) == 1)
            q   = shiftdim( q, 1 );
        end
    end % quaternion.eulerangles

    function q = eye( N )
% function q = eye( N )
        if nargin < 1
            N   = 1;
        end
        if isempty(N) || (N <= 0)
            q   = quaternion.empty;
        else
            q   = quaternion( eye(N), 0, 0, 0 );
        end
    end % quaternion.eye

    function q = nan( varargin )
% function q = quaternion.nan( siz )
        if isempty( varargin )
            siz = [1 1];
        elseif numel( varargin ) > 1
            siz = [varargin{:}];
        elseif isempty( varargin{1} )
            siz = [0 0];
        elseif numel( varargin{1} ) > 1
            siz = varargin{1};
        else
            siz = [varargin{1} varargin{1}];
        end
        if prod( siz ) == 0
            q   = reshape( quaternion.empty, siz );
        else
            q   = quaternion( nan(siz), nan, nan, nan );
        end
    end % quaternion.nan

    function q = NaN( varargin )
% function q = quaternion.NaN( siz )
        q   = quaternion.nan( varargin{:} );
    end % quaternion.NaN

    function q = ones( varargin )
% function q = quaternion.ones( siz )
        if isempty( varargin )
            siz = [1 1];
        elseif numel( varargin ) > 1
            siz = [varargin{:}];
        elseif isempty( varargin{1} )
            siz = [0 0];
        elseif numel( varargin{1} ) > 1
            siz = varargin{1};
        else
            siz = [varargin{1} varargin{1}];
        end
        if prod( siz ) == 0
            q   = reshape( quaternion.empty, siz );
        else
            q   = quaternion( ones(siz), 0, 0, 0 );
        end
    end % quaternion.ones

    function q = rand( varargin )
% function q = quaternion.rand( siz )
% Input:
%  siz      size of output array q
% Output:
%  q        uniform random quaternions, NOT normalized to 1,
%           0 <= q.e(1) <= 1, -1 <= q.e(2:4) <= 1
        if isempty( varargin )
            siz = [1 1];
        elseif numel( varargin ) > 1
            siz = [varargin{:}];
        elseif isempty( varargin{1} )
            siz = [0 0];
        elseif numel( varargin{1} ) > 1
            siz = varargin{1};
        else
            siz = [varargin{1} varargin{1}];
        end
        if prod( siz ) == 0
            q   = quaternion.empty;
            return;
        end
        d   = [ rand( [1, siz] ); 2 * rand( [3, siz] ) - 1 ];
        q   = quaternion( d(1,:), d(2,:), d(3,:), d(4,:) );
        q   = reshape( q, siz );
    end % quaternion.rand

    function q = randRot( varargin )
% function q = quaternion.randRot( siz )
% Random quaternions uniform in rotation space
% Input:
%  siz      size of output array q
% Output:
%  q        random quaternions, normalized to 1, 0 <= q.e(1) <= 1,
%           uniform over the 3D surface of a 4 dimensional hypersphere
        if isempty( varargin )
            siz = [1 1];
        elseif numel( varargin ) > 1
            siz = [varargin{:}];
        elseif isempty( varargin{1} )
            siz = [0 0];
        elseif numel( varargin{1} ) > 1
            siz = varargin{1};
        else
            siz = [varargin{1} varargin{1}];
        end
        if prod( siz ) == 0
            q   = quaternion.empty;
            return;
        end
        d   = randn( [4, prod( siz )] );
        n   = sqrt( sum( d.^2, 1 ));
        dn  = bsxfun( @rdivide, d, n );
        neg = dn(1,:) < 0;
        dn(:,neg) = -dn(:,neg);
        q   = quaternion( dn(1,:), dn(2,:), dn(3,:), dn(4,:) );
        q   = reshape( q, siz );
    end % quaternion.randRot

    function q = rotateutov( u, v, dimu, dimv )
% function q = quaternion.rotateutov( u, v, dimu, dimv )
% Construct quaternions to rotate vectors u into directions of vectors v
% Inputs:
%  u        3x1 or 3xN or 1x3 or Nx3 arrays of vectors
%  v        3x1 or 3xN or 1x3 or Nx3 arrays of vectors
%  dimu [OPTIONAL] dimension of u with size 3 to use
%  dimv [OPTIONAL] dimension of v with size 3 to use
% Output:
%  q        quaternion array
        if (nargin < 3) || isempty( dimu )
            [u, dimu, permu] = finddim( u, 3 );
            if dimu == 0
                error( 'u must have a dimension of size 3' );
            end
        elseif dimu > 1
            ndmu  = ndims( u );
            permu = [ dimu : ndmu, 1 : dimu-1 ];
            u     = permute( u, permu );
        else
            permu = 1 : ndims(u);
        end
        siu     = size( u );
        siu(1)  = 1;
        neu     = prod( siu );
        if (nargin < 4) || isempty( dimv )
            [v, dimv, permv] = finddim( v, 3 );
            if dimv == 0
                error( 'v must have a dimension of size 3' );
            end
        elseif dimv > 1
            ndmv  = ndims( v );
            permv = [ dimv : ndmv, 1 : dimv-1 ];
            v     = permute( v, permv );
        else
            permv = 1 : ndims(v);
        end
        siv     = size( v );
        siv(1)  = 1;
        nev     = prod( siv );
        if neu == nev
            siz  = siu;
            nel  = neu;
            perm = permu;
            dim  = dimu;
        elseif (neu > 1) && (nev == 1)
            siz  = siu;
            nel  = neu;
            perm = permu;
            dim  = dimu;
        elseif (neu == 1) && (nev > 1)
            siz  = siv;
            nel  = nev;
            perm = permv;
            dim  = dimv;
        else
            error( 'Number of 3 element vectors in u and v must be 1 or equal' );
        end
        for iel = nel : -1 : 1
            q(iel)  = UV2q( u(:,min(iel,neu)), v(:,min(iel,nev)) );
        end
        if dim > 1
            q   = ipermute( reshape( q, siz ), perm );
        end
    end % quaternion.rotateutov

    function q = rotationmatrix( R )
% function q = quaternion.rotationmatrix( R )
% Construct quaternions from rotation (or direction cosine) matrices
% Input:
%  R        3x3xN rotation (or direction cosine) matrices
% Output:
%  q        quaternion array
        siz = [size(R) 1 1];
        if ~all( siz(1:2) == [3 3] ) || ...
           (abs( det( R(:,:,1) ) - 1 ) > eps(16) )
            error( 'Rotation matrices must be 3x3xN with det(R) == 1' );
        end
        nel = prod( siz(3:end) );
        for iel = nel : -1 : 1
            d(:,iel) = RotMat2e( chop( R(:,:,iel) ));
        end
        q   = quaternion( d(1,:), d(2,:), d(3,:), d(4,:) );
        q   = normalize( q );
        q   = reshape( q, siz(3:end) );
    end % quaternion.rotationmatrix

    function q = zeros( varargin )
% function q = quaternion.zeros( siz )
        if isempty( varargin )
            siz = [1 1];
        elseif numel( varargin ) > 1
            siz = [varargin{:}];
        elseif isempty( varargin{1} )
            siz = [0 0];
        elseif numel( varargin{1} ) > 1
            siz = varargin{1};
        else
            siz = [varargin{1} varargin{1}];
        end
        if prod( siz ) == 0
            q   = reshape( quaternion.empty, siz );
        else
            q   = quaternion( zeros(siz), 0, 0, 0 );
        end
    end % quaternion.zeros

end % methods(Static)
end % classdef quaternion

% Scalar rotation conversion functions
function eout = AngAxis2e( angle, axis )
% function eout = AngAxis2e( angle, axis )
% One Angle-Axis -> one quaternion
s   = sin( 0.5 * angle );
v   = axis(:);
vn  = norm( v );
if vn == 0
    if s == 0
        c   = 0;
    else
        c   = 1;
    end
    u   = zeros( 3, 1 );
else
    c   = cos( 0.5 * angle );
    u   = v(:) ./ vn;
end
eout    = [ c; s * u ];
if (eout(1) < 0) && (mod( angle/(2*pi), 2 ) ~= 1)
    eout = -eout; % rotationally equivalent quaternion with real element >= 0
end
end % AngAxis2e

function qout = EulerAng2q( axes, angles )
% function qout = EulerAng2q( axes, angles )
% One triplet Euler Angles -> one quaternion
na   = length( axes );
axis = zeros( 3, na );
for i0 = 1 : na
    switch axes(i0)
        case {'1', 'i', 'x', 'X'}
            axis(:,i0) = [ 1; 0; 0 ];
        case {'2', 'j', 'y', 'Y'}
            axis(:,i0) = [ 0; 1; 0 ];
        case {'3', 'k', 'z', 'Z'}
            axis(:,i0) = [ 0; 0; 1 ];
        otherwise
            error( 'Illegal axis designation' );
    end
end
q0   = quaternion.angleaxis( angles(:).', axis );
qout = q0(1);
for i0 = 2 : numel(q0)
    qout = product( q0(i0), qout );
end
if qout.e(1) < 0
    qout = -qout; % rotationally equivalent quaternion with real element >= 0
end
end % EulerAng2q

function eout = RotMat2e( R )
% function eout = RotMat2e( R )
% One Rotation Matrix -> one quaternion
eout    = zeros(4,1);
if ~all( all( R == 0 ))
    eout(1) = 0.5 * sqrt( max( 0, R(1,1) + R(2,2) + R(3,3) + 1 ));
    if eout(1) == 0
        eout(2) = sqrt( max( 0, -0.5 *( R(2,2) + R(3,3) ))) * ...
                        sgn( -R(2,3) );
        eout(3) = sqrt( max( 0, -0.5 *( R(1,1) + R(3,3) ))) * ...
                        sgn( -R(1,3) );
        eout(4) = sqrt( max( 0, -0.5 *( R(1,1) + R(2,2) ))) * ...
                        sgn( -R(1,2) );
    else
        eout(2) = 0.25 *( R(3,2) - R(2,3) )/ eout(1);
        eout(3) = 0.25 *( R(1,3) - R(3,1) )/ eout(1);
        eout(4) = 0.25 *( R(2,1) - R(1,2) )/ eout(1);
    end
end
end % RotMat2e

function qout = UV2q( u, v )
% function qout = UV2q( u, v )
% One pair vectors U, V -> one quaternion
w       = cross( u, v );    % construct vector w perpendicular to u and v
magw    = norm( w );
dotuv   = dot( u, v );
if magw == 0
% Either norm(u) == 0 or norm(v) == 0 or dotuv/(norm(u)*norm(v)) == 1
    if dotuv >= 0
        qout    = quaternion( 1, 0, 0, 0 );
        return;
    end
% dotuv/(norm(u)*norm(v)) == -1
% If v == [v(1); 0; 0], rotate by pi about the [0; 0; 1] axis
    if (v(2) == 0) && (v(3) == 0)
        qout    = quaternion( 0, 0, 0, 1 );
        return;
    end
% Otherwise constuct "what" such that dot(v,what) == 0, and rotate about it
% by pi
    what    = [ 0; -v(3); v(2) ]./ sqrt( v(2)^2 + v(3)^2 );
    costh   = -1;
else
% Use w as rotation axis, angle between u and v as rotation angle
    what    = w(:) / magw;
    costh   = dotuv /( norm(u) * norm(v) );
end
c       = sqrt( 0.5 *( 1 + costh ));    % real element >= 0
s       = sqrt( 0.5 *( 1 - costh ));
eout    = [ c; s * what ];
qout    = quaternion( eout(1), eout(2), eout(3), eout(4) );
end % UV2q

% Helper functions
function out = chop( in, tol )
% function out = chop( in, tol )
% Replace values that differ from an integer by <= tol by the integer
% Inputs:
%  in       input array
%  tol      tolerance, default = eps
% Output:
%  out      input array with integer replacements, if any
if (nargin < 2) || isempty( tol )
    tol = eps;
end
out = in;
rin = round( in );
lx  = abs( rin - in ) <= tol;
out(lx) = rin(lx);
end % chop

function [aout, dim, perm] = finddim( ain, len )
% function [aout, dim, perm] = finddim( ain, len )
% Find first dimension in ain of length len, permute ain to make it first
% Inputs:
%  ain(s1,s2,...)   data array, size = [s1, s2, ...]
%  len              length sought, e.g. s2 == len
%                   if len < 0, then find first dimension >= |len|
% Outputs:
%  aout(s2,...,s1)  data array, permuted so first dimension is length len
%  dim              dimension number of length len, 0 if ain has none
%  perm             permutation order (for permute and ipermute) of aout,
%                   e.g. [2, ..., 1]
% Notes: if no dimension has length len, aout = ain, dim = 0, perm = 1:ndm
%        ain = ipermute( aout, perm )
siz  = size( ain );
ndm  = length( siz );
if len < 0
    dim  = find( siz >= -len, 1, 'first' );
else
    dim  = find( siz == len, 1, 'first' );
end
if isempty( dim )
    dim  = 0;
end
if dim < 2
    aout = ain;
    perm = 1 : ndm;
else
% Permute so that dim becomes the first dimension
    perm = [ dim : ndm, 1 : dim-1 ];
    aout = permute( ain, perm );
end
end % finddim

function s = sgn( x )
% function s = sgn( x ), if x >= 0, s = 1, else s = -1
s   = ones( size( x ));
s(x < 0) = -1;
end % sgn

function [u, n] = unitvector( v, dim )
% function [u, n] = unitvector( v, dim )
% Inputs:
%  v                matrix of vectors
%  dim [OPTIONAL]   dimension to normalize, dim >= 1
%                   if no dim input, use first dimension of length >= 2
% Outputs:
%  u                matrix of unit vectors (except for vectors of norm 0)
%  n                matrix same size as v and u of norms
ndm = ndims( v );
if (nargin < 2) || isempty( dim )
    [v, dim, perm] = finddim( v, -2 );
    if dim == 0
        n     = sqrt( v.*conj(v) );
        n0    = (n ~= 0) & (n ~= 1);
        u     = v;
        u(n0) = v(n0) ./ n(n0);
        return;
    end
else
    perm = [ dim : ndm, 1 : dim-1 ];
    v    = permute( v, perm );
end
u     = v;
sv    = size( v );
n     = repmat( sqrt( sum( v.*conj(v), 1 )), [sv(1) ones(1,ndm-1)] );
n0    = (n ~= 0) & (n ~= 1);
u(n0) = v(n0) ./ n(n0);
u     = ipermute( u, perm );
if nargout > 1
    n = ipermute( n, perm );
end
end % unitvector