	.section	__TEXT,__text,regular,pure_instructions
	.build_version macos, 11, 0	sdk_version 11, 3
	.globl	__Z13copy_baselinePKiPii        ; -- Begin function _Z13copy_baselinePKiPii
	.p2align	2
__Z13copy_baselinePKiPii:               ; @_Z13copy_baselinePKiPii
	.cfi_startproc
; %bb.0:
	cmp	w2, #1                          ; =1
	b.lt	LBB0_9
; %bb.1:
	mov	w8, w2
	cmp	w2, #16                         ; =16
	b.lo	LBB0_6
; %bb.2:
	lsl	x9, x8, #2
	add	x10, x1, x9
	add	x9, x0, x9
	cmp	x9, x1
	ccmp	x10, x0, #0, hi
	b.hi	LBB0_6
; %bb.3:
	and	x9, x8, #0xfffffff0
	add	x10, x1, #32                    ; =32
	add	x11, x0, #32                    ; =32
	mov	x12, x9
LBB0_4:                                 ; =>This Inner Loop Header: Depth=1
	ldp	q0, q1, [x11, #-32]
	ldp	q2, q3, [x11], #64
	stp	q0, q1, [x10, #-32]
	stp	q2, q3, [x10], #64
	subs	x12, x12, #16                   ; =16
	b.ne	LBB0_4
; %bb.5:
	cmp	x9, x8
	b.ne	LBB0_7
	b	LBB0_9
LBB0_6:
	mov	x9, #0
LBB0_7:
	lsl	x11, x9, #2
	add	x10, x1, x11
	add	x11, x0, x11
	sub	x8, x8, x9
LBB0_8:                                 ; =>This Inner Loop Header: Depth=1
	ldr	w9, [x11], #4
	str	w9, [x10], #4
	subs	x8, x8, #1                      ; =1
	b.ne	LBB0_8
LBB0_9:
	ret
	.cfi_endproc
                                        ; -- End function
	.globl	__Z14copy_baseline2PKiPii       ; -- Begin function _Z14copy_baseline2PKiPii
	.p2align	2
__Z14copy_baseline2PKiPii:              ; @_Z14copy_baseline2PKiPii
	.cfi_startproc
; %bb.0:
                                        ; kill: def $w2 killed $w2 def $x2
	subs	w9, w2, #1                      ; =1
	b.lt	LBB1_8
; %bb.1:
	and	w8, w2, #0x7
	cmp	w9, #7                          ; =7
	b.hs	LBB1_3
; %bb.2:
	mov	x9, #0
	b	LBB1_5
LBB1_3:
	mov	x9, #0
	and	x10, x2, #0xfffffff8
	add	x11, x1, #16                    ; =16
	add	x12, x0, #16                    ; =16
LBB1_4:                                 ; =>This Inner Loop Header: Depth=1
	ldur	w13, [x12, #-16]
	stur	w13, [x11, #-16]
	ldur	w13, [x12, #-12]
	stur	w13, [x11, #-12]
	ldur	w13, [x12, #-8]
	stur	w13, [x11, #-8]
	ldur	w13, [x12, #-4]
	stur	w13, [x11, #-4]
	ldr	w13, [x12]
	str	w13, [x11]
	ldr	w13, [x12, #4]
	str	w13, [x11, #4]
	ldr	w13, [x12, #8]
	str	w13, [x11, #8]
	ldr	w13, [x12, #12]
	str	w13, [x11, #12]
	add	x9, x9, #8                      ; =8
	add	x11, x11, #32                   ; =32
	add	x12, x12, #32                   ; =32
	cmp	w10, w9
	b.ne	LBB1_4
LBB1_5:
	cbz	w8, LBB1_8
; %bb.6:
	lsl	x10, x9, #2
	add	x9, x1, x10
	add	x10, x0, x10
LBB1_7:                                 ; =>This Inner Loop Header: Depth=1
	ldr	w11, [x10], #4
	str	w11, [x9], #4
	subs	w8, w8, #1                      ; =1
	b.ne	LBB1_7
LBB1_8:
	ret
	.cfi_endproc
                                        ; -- End function
.subsections_via_symbols
