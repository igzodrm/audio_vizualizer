using NAudio.Dsp;
using NAudio.Wave;
using OpenTK.Compute.OpenCL;
using OpenTK.Graphics.OpenGL4;
using OpenTK.Mathematics;
using OpenTK.Windowing.Common;
using OpenTK.Windowing.Desktop;
using OpenTK.Windowing.GraphicsLibraryFramework;
using System;
using System.Collections.Concurrent;
using System.Diagnostics;
using System.Drawing;
using System.Drawing.Imaging;
using System.Drawing.Text;
using System.Runtime.InteropServices;
using GLPixelFormat = OpenTK.Graphics.OpenGL4.PixelFormat;
using SDPixelFormat = System.Drawing.Imaging.PixelFormat;


internal static class Program
{
    [DllImport("user32.dll", CharSet = CharSet.Unicode)]
    private static extern int MessageBoxW(IntPtr hWnd, string text, string caption, uint type);

    [STAThread]
    private static void Main()
    {
        try
        {
            var game = GameWindowSettings.Default;
            game.UpdateFrequency = 240;

            var native = new NativeWindowSettings
            {
                ClientSize = new Vector2i(1280, 720),
                Title = "AudioViz | Space: start/stop | 1-5 layers | H: fullscreen | U: hide UI | Esc: exit",
                WindowBorder = WindowBorder.Resizable,
                StartVisible = true,
                APIVersion = new Version(3, 3),
                Profile = ContextProfile.Core,
                Flags = ContextFlags.ForwardCompatible,
            };

            using var window = new VisualizerWindow(game, native);
            window.Run();
        }
        catch (Exception ex)
        {
            MessageBoxW(IntPtr.Zero, ex.ToString(), "AudioViz crash", 0);
        }
    }
}

#region Window / Rendering

internal sealed class VisualizerWindow : GameWindow
{
    // ====== Tweakables ======
    private const int BarsCount = 420;          // 300..500
    private const int WaveSize = 2048;
    // oscilloscope samples
    private const int FftSize = 2048;           // 1024 or 2048 (latency vs detail)
    private const float MinFreq = 20f;
    private const float MaxFreq = 20000f;
    private const float SpectrumGapDegrees = 40f;

    private bool _hideUi = false;

    // Spectrum geometry
    private float _innerRadius;
    private float _barMaxLen;
    private float _barThickness;

    private float _lastBeatStrength = 0f;
    private Vector4 _lastBeatColor = new(0.085f, 0.035f, 0.12f, 1f);

    // ===== Text labels =====
    private ShaderProgram? _shaderText;
    private int _vaoText, _vboText;

    [StructLayout(LayoutKind.Sequential)]
    private struct TextVertex
    {
        public Vector2 Pos;
        public Vector2 UV;
        public TextVertex(Vector2 p, Vector2 uv) { Pos = p; UV = uv; }
        public const int SizeInBytes = (2 + 2) * 4;
    }

    private struct TextLabel
    {
        public int Tex;
        public int W;
        public int H;
        public string Text;
    }

    private TextLabel _lblSpectrum, _lblStereo, _lblWave;
    private Vector2 _lblSpectrumPos, _lblStereoPos, _lblWavePos;
    private TextVertex[] _textQuad = new TextVertex[6];
    private int _labelFontPx = -1;


    // Layout
    private Vector2 _center;
    private float _waveBaseY;
    private float _waveAmp;
    private float _waveSideMargin;
    private float _waveBandTop; // верхняя граница зоны осциллограммы

    // State toggles
    private bool _vizEnabled = true;
    private bool _cleanMode = false;
    private bool _layerSpectrum = true;
    private bool _layerWave = true;
    private bool _layerParticles = true;
    private bool _layerBackground = true;

    // Audio / analysis
    private readonly AudioAnalyzer _audio = new(BarsCount, WaveSize, FftSize, MinFreq, MaxFreq);
    private readonly float[] _barTargets = new float[BarsCount];
    private readonly float[] _barSmoothed = new float[BarsCount];
    private readonly float[] _wave = new float[WaveSize];

    private const int StereoPoints = 1024;

    private bool _layerStereo = true;

    private readonly float[] _stL = new float[StereoPoints];
    private readonly float[] _stR = new float[StereoPoints];

    private Vertex[] _stereoFrameVerts = Array.Empty<Vertex>();
    private Vertex[] _stereoScopeVerts = Array.Empty<Vertex>();

    private float _stereoX, _stereoY, _stereoSize;


    private double _waveRefreshAcc = 0.0;

    // Particles
    private readonly ParticleSystem _particles = new(maxParticles: 4000);

    // GL: shaders
    private ShaderProgram? _shaderColor;
    private ShaderProgram? _shaderBackground;

    // GL: VAO/VBO for dynamic geometry
    private int _vaoDynamic, _vboDynamic;
    private int _vaoBg, _vboBg;

    // CPU vertex buffers (reused)
    private Vertex[] _spectrumVerts = Array.Empty<Vertex>();
    private Vertex[] _waveVerts = Array.Empty<Vertex>();
    private Vertex[] _particleVerts = Array.Empty<Vertex>();
    private Vertex[] _circleVerts = Array.Empty<Vertex>();

    private double _timeSec;
    private double _lastBeatFlash; // for subtle bg emphasis

    private readonly float[] _tmpL = new float[WaveSize];
    private readonly float[] _tmpR = new float[WaveSize];

    public VisualizerWindow(GameWindowSettings gameWindowSettings, NativeWindowSettings nativeWindowSettings)
        : base(gameWindowSettings, nativeWindowSettings) { }

    protected override void OnLoad()
    {
        base.OnLoad();
        VSync = VSyncMode.Off;

        GL.ClearColor(0f, 0f, 0f, 1f);
        GL.Disable(EnableCap.DepthTest);
        GL.Enable(EnableCap.Blend);
        GL.BlendFunc(BlendingFactor.SrcAlpha, BlendingFactor.OneMinusSrcAlpha);

        _shaderColor = new ShaderProgram(Shaders.ColorVert, Shaders.ColorFrag);
        _shaderBackground = new ShaderProgram(Shaders.BgVert, Shaders.BgFrag);

        _shaderText = new ShaderProgram(Shaders.TextVert, Shaders.TextFrag);

        // VAO/VBO for text quads
        _vaoText = GL.GenVertexArray();
        _vboText = GL.GenBuffer();

        GL.BindVertexArray(_vaoText);
        GL.BindBuffer(BufferTarget.ArrayBuffer, _vboText);
        GL.BufferData(BufferTarget.ArrayBuffer, TextVertex.SizeInBytes * 6, IntPtr.Zero, BufferUsageHint.StreamDraw);

        GL.EnableVertexAttribArray(0);
        GL.VertexAttribPointer(0, 2, VertexAttribPointerType.Float, false, TextVertex.SizeInBytes, 0);

        GL.EnableVertexAttribArray(1);
        GL.VertexAttribPointer(1, 2, VertexAttribPointerType.Float, false, TextVertex.SizeInBytes, 2 * sizeof(float));

        GL.BindVertexArray(0);

        // Dynamic VAO/VBO for spectrum/wave/particles/circle
        _vaoDynamic = GL.GenVertexArray();
        _vboDynamic = GL.GenBuffer();
        GL.BindVertexArray(_vaoDynamic);
        GL.BindBuffer(BufferTarget.ArrayBuffer, _vboDynamic);

        // Allocate a “big enough” buffer once; we’ll use BufferSubData for actual content
        GL.BufferData(BufferTarget.ArrayBuffer, Vertex.SizeInBytes * 100000, IntPtr.Zero, BufferUsageHint.StreamDraw);

        GL.EnableVertexAttribArray(0);
        GL.VertexAttribPointer(0, 2, VertexAttribPointerType.Float, false, Vertex.SizeInBytes, 0);

        GL.EnableVertexAttribArray(1);
        GL.VertexAttribPointer(1, 4, VertexAttribPointerType.Float, false, Vertex.SizeInBytes, 2 * sizeof(float));

        GL.BindVertexArray(0);

        // Background quad VAO/VBO
        _vaoBg = GL.GenVertexArray();
        _vboBg = GL.GenBuffer();
        GL.BindVertexArray(_vaoBg);
        GL.BindBuffer(BufferTarget.ArrayBuffer, _vboBg);

        // Fullscreen quad in NDC
        float[] bg = {
            -1f,-1f,  1f,-1f,  1f, 1f,
            -1f,-1f,  1f, 1f, -1f, 1f
        };
        GL.BufferData(BufferTarget.ArrayBuffer, bg.Length * sizeof(float), bg, BufferUsageHint.StaticDraw);
        GL.EnableVertexAttribArray(0);
        GL.VertexAttribPointer(0, 2, VertexAttribPointerType.Float, false, 2 * sizeof(float), 0);
        GL.BindVertexArray(0);

        RecomputeLayout();

        _audio.Start();

        // Beat events come from audio thread -> queue -> consumed in Update
        _audio.OnBeat += beat =>
        {
            // Keep it lightweight: just enqueue already done inside analyzer,
            // but this hook is available if needed.
        };
    }

    private static int CreateTextTexture(string text, int fontPx, out int w, out int h)
    {
        using var font = new Font("Segoe UI", fontPx, FontStyle.Bold, GraphicsUnit.Pixel);

        // сначала измерим
        using var tmpBmp = new Bitmap(8, 8, SDPixelFormat.Format32bppArgb);
        using var g0 = Graphics.FromImage(tmpBmp);
        g0.TextRenderingHint = TextRenderingHint.AntiAliasGridFit;

        var size = g0.MeasureString(text, font);
        w = (int)MathF.Ceiling(size.Width) + 12;
        h = (int)MathF.Ceiling(size.Height) + 8;

        using var bmp = new Bitmap(w, h, SDPixelFormat.Format32bppArgb);
        using var g = Graphics.FromImage(bmp);
        g.Clear(Color.Transparent);
        g.TextRenderingHint = TextRenderingHint.AntiAliasGridFit;

        using var brush = new SolidBrush(Color.FromArgb(255, 255, 255, 255));
        g.DrawString(text, font, brush, new PointF(6, 3));

        int tex = GL.GenTexture();
        GL.BindTexture(TextureTarget.Texture2D, tex);

        var data = bmp.LockBits(new Rectangle(0, 0, w, h), ImageLockMode.ReadOnly, SDPixelFormat.Format32bppArgb);
        GL.TexImage2D(TextureTarget.Texture2D, 0, PixelInternalFormat.Rgba, w, h, 0,
              GLPixelFormat.Bgra, PixelType.UnsignedByte, data.Scan0);
        bmp.UnlockBits(data);

        GL.TexParameter(TextureTarget.Texture2D, TextureParameterName.TextureMinFilter, (int)TextureMinFilter.Linear);
        GL.TexParameter(TextureTarget.Texture2D, TextureParameterName.TextureMagFilter, (int)TextureMagFilter.Linear);
        GL.TexParameter(TextureTarget.Texture2D, TextureParameterName.TextureWrapS, (int)TextureWrapMode.ClampToEdge);
        GL.TexParameter(TextureTarget.Texture2D, TextureParameterName.TextureWrapT, (int)TextureWrapMode.ClampToEdge);

        GL.BindTexture(TextureTarget.Texture2D, 0);
        return tex;
    }

    private void EnsureLabels()
    {
        // шрифт масштабируется от блока, поэтому в fullscreen выглядит так же “близко”
        int fontPx = (int)Math.Clamp(_stereoSize * 0.11f, 14f, 22f);

        if (fontPx == _labelFontPx && _lblSpectrum.Tex != 0) return;
        _labelFontPx = fontPx;

        // удалить старые текстуры
        void Kill(ref TextLabel lbl)
        {
            if (lbl.Tex != 0) GL.DeleteTexture(lbl.Tex);
            lbl = default;
        }

        Kill(ref _lblSpectrum);
        Kill(ref _lblStereo);
        Kill(ref _lblWave);

        _lblSpectrum = new TextLabel { Text = "SPECTRUM" };
        _lblStereo = new TextLabel { Text = "STEREO WIDTH" };
        _lblWave = new TextLabel { Text = "OSCILLOSCOPE" };

        _lblSpectrum.Tex = CreateTextTexture(_lblSpectrum.Text, fontPx, out _lblSpectrum.W, out _lblSpectrum.H);
        _lblStereo.Tex = CreateTextTexture(_lblStereo.Text, fontPx, out _lblStereo.W, out _lblStereo.H);
        _lblWave.Tex = CreateTextTexture(_lblWave.Text, fontPx, out _lblWave.W, out _lblWave.H);
    }

    private static float ClampX(float x, float w, int labelW)
    {
        float pad = 8f;
        return MathF.Min(MathF.Max(x, pad), w - labelW - pad);
    }

    private static float ClampY(float y, float h, int labelH)
    {
        float pad = 8f;
        return MathF.Min(MathF.Max(y, pad), h - labelH - pad);
    }

    private void UpdateLabelPositions()
    {
        float w = Size.X;
        float h = Size.Y;

        // Верх спектра (по внешнему радиусу: inner + barMaxLen)
        float outerR = _innerRadius + _barMaxLen;
        float spectrumTopY = _center.Y - outerR;

        // Верх stereo-квадрата
        float stereoTopY = _stereoY;

        // Берём "самый верхний" из блоков (меньший Y), чтобы обе подписи гарантированно были над обоими
        float topObjY = MathF.Min(stereoTopY, spectrumTopY);

        // Хотим, чтобы НИЗ текста был на одной линии и близко к блокам
        float gapToBlock = 1f;           // ближе к блокам (поставь 2f если хочется ещё ближе)
        float yBottomCommon = topObjY - gapToBlock;

        // --- STEREO WIDTH: левый край stereo, выравниваем по общему низу ---
        float stereoTextY = yBottomCommon - _lblStereo.H;
        _lblStereoPos = new Vector2(
            ClampX(_stereoX, w, _lblStereo.W),
            ClampY(stereoTextY, h, _lblStereo.H)
        );

        // --- SPECTRUM: по центру спектра, выравниваем по общему низу ---
        float spectrumTextX = _center.X - _lblSpectrum.W * 0.5f;
        float spectrumTextY = yBottomCommon - _lblSpectrum.H;
        _lblSpectrumPos = new Vector2(
            ClampX(spectrumTextX, w, _lblSpectrum.W),
            ClampY(spectrumTextY, h, _lblSpectrum.H)
        );

        // --- OSCILLOSCOPE: как раньше (в верхней части нижней зоны слева) ---
        float waveX = _waveSideMargin;
        float waveY = _waveBandTop + 10f;
        _lblWavePos = new Vector2(
            ClampX(waveX, w, _lblWave.W),
            ClampY(waveY, h, _lblWave.H)
        );
    }


    private void DrawLabel(in TextLabel lbl, Vector2 pos, Vector4 tint)
    {
        if (_shaderText is null || lbl.Tex == 0) return;

        float x = pos.X;
        float y = pos.Y;
        float w = lbl.W;
        float h = lbl.H;

        // quad (2 triangles)
        _textQuad[0] = new TextVertex(new Vector2(x, y), new Vector2(0, 0));
        _textQuad[1] = new TextVertex(new Vector2(x + w, y), new Vector2(1, 0));
        _textQuad[2] = new TextVertex(new Vector2(x + w, y + h), new Vector2(1, 1));
        _textQuad[3] = new TextVertex(new Vector2(x, y), new Vector2(0, 0));
        _textQuad[4] = new TextVertex(new Vector2(x + w, y + h), new Vector2(1, 1));
        _textQuad[5] = new TextVertex(new Vector2(x, y + h), new Vector2(0, 1));

        _shaderText.Use();
        _shaderText.SetUniform("uResolution", new Vector2(Size.X, Size.Y));
        _shaderText.SetUniform("uTint", tint);

        GL.ActiveTexture(TextureUnit.Texture0);
        GL.BindTexture(TextureTarget.Texture2D, lbl.Tex);
        _shaderText.SetUniform("uTex", 0);

        GL.BindVertexArray(_vaoText);
        GL.BindBuffer(BufferTarget.ArrayBuffer, _vboText);
        GL.BufferSubData(BufferTarget.ArrayBuffer, IntPtr.Zero, TextVertex.SizeInBytes * 6, _textQuad);

        GL.DrawArrays(PrimitiveType.Triangles, 0, 6);

        GL.BindVertexArray(0);
        GL.BindTexture(TextureTarget.Texture2D, 0);
    }

    protected override void OnUnload()
    {
        base.OnUnload();

        _audio.Dispose();

        if (_shaderColor is not null) _shaderColor.Dispose();
        if (_shaderBackground is not null) _shaderBackground.Dispose();

        if (_vboDynamic != 0) GL.DeleteBuffer(_vboDynamic);
        if (_vaoDynamic != 0) GL.DeleteVertexArray(_vaoDynamic);
        if (_vboBg != 0) GL.DeleteBuffer(_vboBg);
        if (_vaoBg != 0) GL.DeleteVertexArray(_vaoBg);
        if (_lblSpectrum.Tex != 0) GL.DeleteTexture(_lblSpectrum.Tex);
        if (_lblStereo.Tex != 0) GL.DeleteTexture(_lblStereo.Tex);
        if (_lblWave.Tex != 0) GL.DeleteTexture(_lblWave.Tex);

        if (_vboText != 0) GL.DeleteBuffer(_vboText);
        if (_vaoText != 0) GL.DeleteVertexArray(_vaoText);

        _shaderText?.Dispose();
    }

    protected override void OnResize(ResizeEventArgs e)
    {
        base.OnResize(e);
        GL.Viewport(0, 0, Size.X, Size.Y);
        RecomputeLayout();
    }

    private void BuildStereoVertices()
    {
        // Рамка + диагонали + крест в центре
        float x0 = _stereoX;
        float y0 = _stereoY;
        float x1 = _stereoX + _stereoSize;
        float y1 = _stereoY + _stereoSize;
        float cx = (x0 + x1) * 0.5f;
        float cy = (y0 + y1) * 0.5f;

        var frameCol = new Vector4(0.35f, 0.35f, 0.40f, 0.35f);

        int v = 0;

        void Seg(float ax, float ay, float bx, float by)
        {
            _stereoFrameVerts[v++] = new Vertex(new Vector2(ax, ay), frameCol);
            _stereoFrameVerts[v++] = new Vertex(new Vector2(bx, by), frameCol);
        }

        // border (4 segments)
        Seg(x0, y0, x1, y0);
        Seg(x1, y0, x1, y1);
        Seg(x1, y1, x0, y1);
        Seg(x0, y1, x0, y0);

        // diagonals
        Seg(x0, y1, x1, y0);
        Seg(x0, y0, x1, y1);

        // cross center
        Seg(cx, y0, cx, y1);
        Seg(x0, cy, x1, cy);

        // Scope line (L vs R)
        // нормализация по максимуму, чтобы фигура заполняла область
        float maxAbs = 1e-3f;
        for (int i = 0; i < StereoPoints; i++)
        {
            float a = MathF.Abs(_stL[i]);
            float b = MathF.Abs(_stR[i]);
            if (a > maxAbs) maxAbs = a;
            if (b > maxAbs) maxAbs = b;
        }
        float scale = 0.90f / maxAbs;

        var scopeCol = new Vector4(0.95f, 0.95f, 1.00f, 0.75f);

        for (int i = 0; i < StereoPoints; i++)
        {
            float L = _stL[i] * scale;
            float R = _stR[i] * scale;

            float px = cx + L * (_stereoSize * 0.5f);
            float py = cy - R * (_stereoSize * 0.5f); // R вверх

            // лёгкое “старение” хвоста
            float t = i / (float)(StereoPoints - 1);
            var c = scopeCol;
            c.W *= (0.15f + 0.85f * t);

            _stereoScopeVerts[i] = new Vertex(new Vector2(px, py), c);
        }
    }

    private void RecomputeLayout()
    {
        float w = Size.X;
        float h = Size.Y;

        // -----------------------------
        // 1) Нижняя зона: осциллограмма (чуть больше места + защита от выхода за экран)
        // -----------------------------
        _waveSideMargin = w * 0.06f;

        // место под осциллограмму
        float waveBandHeight = MathF.Max(190f, h * 0.30f);
        waveBandHeight = MathF.Min(waveBandHeight, h * 0.46f);

        _waveBandTop = h - waveBandHeight;

        // небольшой внутренний отступ, чтобы на пиках не упиралось в низ/верх экрана
        float pad = MathF.Max(6f, waveBandHeight * 0.02f);

        // базовую линию чуть выше (было 0.60)
        _waveBaseY = _waveBandTop + waveBandHeight * 0.58f;

        // желаемая амплитуда (как раньше)
        float desiredAmp = waveBandHeight * 0.42f;

        // ограничиваем амплитуду так, чтобы y оставался в [top+pad .. h-pad]
        float topLimit = _waveBandTop + pad;
        float bottomLimit = h - pad;

        float ampTop = _waveBaseY - topLimit;
        float ampBottom = bottomLimit - _waveBaseY;

        _waveAmp = MathF.Min(desiredAmp, MathF.Max(0f, MathF.Min(ampTop, ampBottom)));

        // -----------------------------
        // 2) Верхняя зона: Stereo + Spectrum
        //    (ближе друг к другу и спектр крупнее)
        // -----------------------------
        float topH = MathF.Max(1f, _waveBandTop);

        float gap = MathF.Max(10f, w * 0.012f);

        float stereoScale = 0.85f;

        float spectrumByHeight = topH * 0.96f;
        float spectrumByWidth = (w - gap) / (1f + stereoScale);

        float spectrumSize = MathF.Min(spectrumByWidth, spectrumByHeight);
        spectrumSize = MathF.Max(spectrumSize, 100f);

        _stereoSize = spectrumSize * stereoScale;

        float groupW = _stereoSize + gap + spectrumSize;
        float left = (w - groupW) * 0.5f;
        if (left < 8f) left = 8f;

        _stereoX = left;
        _stereoY = topH * 0.5f - _stereoSize * 0.5f;

        float spectrumX = _stereoX + _stereoSize + gap;
        float spectrumY = topH * 0.5f - spectrumSize * 0.5f;

        _center = new Vector2(spectrumX + spectrumSize * 0.5f,
                              spectrumY + spectrumSize * 0.5f);

        // -----------------------------
        // 3) Геометрия спектра
        // -----------------------------
        float bound = spectrumSize * 0.5f * 1.02f;
        bound = MathF.Min(bound, spectrumSize * 0.5f * 1.05f);

        _innerRadius = bound * 0.34f;
        _barMaxLen = bound - _innerRadius;

        _barThickness = MathF.Max(1.5f, spectrumSize * 0.0048f);

        // -----------------------------
        // 4) Буферы вершин
        // -----------------------------
        _spectrumVerts = new Vertex[BarsCount * 6];
        _waveVerts = new Vertex[WaveSize];
        _particleVerts = new Vertex[_particles.MaxParticles];
        _circleVerts = new Vertex[1 + 64 + 1];

        _stereoFrameVerts = new Vertex[16];
        _stereoScopeVerts = new Vertex[StereoPoints];

        EnsureLabels();
        UpdateLabelPositions();
        BuildCircleVertices();
    }


    private void BuildCircleVertices()
    {
        Vector2 center = _center;
        float r = _innerRadius * 0.95f;

        // subtle center circle (dark)
        var c = new Vector4(0.03f, 0.03f, 0.04f, 0.65f);
        _circleVerts[0] = new Vertex(center, c);
        int seg = 64;
        for (int i = 0; i <= seg; i++)
        {
            float a = i / (float)seg * MathHelper.TwoPi;
            var p = center + new Vector2(MathF.Cos(a), MathF.Sin(a)) * r;
            _circleVerts[1 + i] = new Vertex(p, c);
        }
    }

    protected override void OnKeyDown(KeyboardKeyEventArgs e)
    {
        base.OnKeyDown(e);

        if (e.Key == Keys.Escape) Close();

        if (e.Key == Keys.Space) _vizEnabled = !_vizEnabled;

        if (e.Key == Keys.D1 || e.Key == Keys.KeyPad1) _layerSpectrum = !_layerSpectrum;
        if (e.Key == Keys.D2 || e.Key == Keys.KeyPad2) _layerWave = !_layerWave;
        if (e.Key == Keys.D3 || e.Key == Keys.KeyPad3) _layerParticles = !_layerParticles;
        if (e.Key == Keys.D4 || e.Key == Keys.KeyPad4) _layerBackground = !_layerBackground;
        if (e.Key == Keys.D5 || e.Key == Keys.KeyPad5) _layerStereo = !_layerStereo;


        if (e.Key == Keys.H)
        {
            _cleanMode = !_cleanMode;
            if (_cleanMode)
            {
                WindowBorder = WindowBorder.Hidden;
                WindowState = WindowState.Fullscreen;
            }
            else
            {
                WindowState = WindowState.Normal;
                WindowBorder = WindowBorder.Resizable;
            }
            RecomputeLayout();
            EnsureLabels();
            UpdateLabelPositions();
        }
        if (e.Key == Keys.U)
            _hideUi = !_hideUi;
    }

    protected override void OnUpdateFrame(FrameEventArgs args)
    {
        base.OnUpdateFrame(args);

        _timeSec += args.Time;

        // Pull audio snapshots
        _audio.CopyBars(_barTargets);
        _audio.CopyWaveform(_wave);

        // Stereo snapshot (L/R)
        if (_layerStereo)
        {
            _audio.CopyStereo(_tmpL, _tmpR);
            int start = WaveSize - StereoPoints;
            for (int i = 0; i < StereoPoints; i++)
            {
                _stL[i] = _tmpL[start + i];
                _stR[i] = _tmpR[start + i];
            }
        }

        // If visualization disabled -> calm down quickly
        // If visualization disabled -> make everything go silent (spectrum + wave + stereo)
        if (!_vizEnabled)
        {
            Array.Clear(_barTargets, 0, _barTargets.Length);
            Array.Clear(_wave, 0, _wave.Length);

            // also freeze stereo scope
            if (_layerStereo)
            {
                Array.Clear(_stL, 0, _stL.Length);
                Array.Clear(_stR, 0, _stR.Length);
            }

            // optional: stop particles instantly on pause (looks cleaner)
            _particles.Clear();
        }

        // Smooth bars: fast attack, slow release (time-based)
        float dt = (float)args.Time;
        // ✅ меньше “визуального лага”
        float attack = 1f - MathF.Exp(-dt / 0.015f);  // было 35ms
        float release = 1f - MathF.Exp(-dt / 0.140f);  // чуть быстрее отпускание

        for (int i = 0; i < BarsCount; i++)
        {
            float t = _barTargets[i];
            float v = _barSmoothed[i];
            float k = (t > v) ? attack : release;
            _barSmoothed[i] = v + (t - v) * k;
        }

        // Consume beat events:
        // ✅ фон реагирует ВСЕГДА (даже если частицы выключены)
        while (_audio.TryDequeueBeat(out var beat))
        {
            _lastBeatFlash = _timeSec;
            _lastBeatStrength = beat.Strength;

            _lastBeatColor = beat.Color;
            _lastBeatColor.W = 1f;

            // частицы — только если включены
            if (_vizEnabled && _layerParticles)
            {
                int count = (int)MathHelper.Lerp(50, 200, beat.Strength);
                _particles.SpawnBurst(_center, count, beat.Strength, beat.Color, Size); // ✅ из центра спектра
            }
        }

        // Update particles
        if (_layerParticles)
            _particles.Update(dt, Size);
        else
            _particles.Clear();
    }


    protected override void OnRenderFrame(FrameEventArgs args)
    {
        base.OnRenderFrame(args);

        GL.Clear(ClearBufferMask.ColorBufferBit);

        if (_shaderColor is null || _shaderBackground is null)
        {
            SwapBuffers();
            return;
        }

        // =========================
        // Background (живой фон)
        // =========================
        if (_layerBackground)
        {
            _shaderBackground.Use();

            float bpm = _audio.BpmSmoothed;
            float speed = MathHelper.Clamp(bpm / 120f, 0.15f, 3.0f);

            // Экспоненциальная огибающая: пульс мягкий и заметный
            float beatEnv = MathF.Exp(-(float)(_timeSec - _lastBeatFlash) / 0.45f);
            float beatGlow = beatEnv * (0.25f + 0.75f * _lastBeatStrength); // сила удара влияет на амплитуду

            _shaderBackground.SetUniform("uTime", (float)_timeSec);
            _shaderBackground.SetUniform("uSpeed", speed);
            _shaderBackground.SetUniform("uBeatGlow", beatGlow);
            _shaderBackground.SetUniform("uBeatTint", _lastBeatColor); // цвет от доминирующей частоты

            GL.BindVertexArray(_vaoBg);
            GL.DrawArrays(PrimitiveType.Triangles, 0, 6);
            GL.BindVertexArray(0);
        }

        // ==========================================
        // Dynamic geometry (Stereo / Spectrum / Wave / Particles)
        // ==========================================
        _shaderColor.Use();
        _shaderColor.SetUniform("uResolution", new Vector2(Size.X, Size.Y));

        GL.BindVertexArray(_vaoDynamic);
        GL.BindBuffer(BufferTarget.ArrayBuffer, _vboDynamic);

        // ---- Stereo L/R ----
        if (_layerStereo && _stereoSize > 5f)
        {
            BuildStereoVertices();

            UploadAndDraw(_stereoFrameVerts, _stereoFrameVerts.Length, PrimitiveType.Lines, lineWidth: 1.5f);

            GL.BlendFunc(BlendingFactor.SrcAlpha, BlendingFactor.One);
            UploadAndDraw(_stereoScopeVerts, _stereoScopeVerts.Length, PrimitiveType.LineStrip, lineWidth: 2.0f);
            GL.BlendFunc(BlendingFactor.SrcAlpha, BlendingFactor.OneMinusSrcAlpha);
        }

        // ---- Center circle behind spectrum ----
        if (_layerSpectrum)
        {
            UploadAndDraw(_circleVerts, _circleVerts.Length, PrimitiveType.TriangleFan);
        }

        // ---- Spectrum bars ----
        if (_layerSpectrum)
        {
            BuildSpectrumVertices();
            UploadAndDraw(_spectrumVerts, _spectrumVerts.Length, PrimitiveType.Triangles);
        }

        // ---- Oscilloscope ----
        if (_layerWave)
        {
            BuildWaveVertices();
            UploadAndDraw(_waveVerts, _waveVerts.Length, PrimitiveType.LineStrip, lineWidth: 2f);
        }

        // ---- Particles ----
        if (_layerParticles)
        {
            int pc = BuildParticleVertices();
            if (pc > 0)
            {
                GL.BlendFunc(BlendingFactor.SrcAlpha, BlendingFactor.One);
                UploadAndDraw(_particleVerts, pc, PrimitiveType.Points, pointSize: 3.5f);
                GL.BlendFunc(BlendingFactor.SrcAlpha, BlendingFactor.OneMinusSrcAlpha);
            }
        }

        GL.BindVertexArray(0);

        // =========================
        // Labels (подписи слоёв)
        // =========================
        bool hideUi = _hideUi;

        if (!hideUi && _shaderText is not null)
        { 
            var tint = new Vector4(0.90f, 0.90f, 1.00f, 0.55f);

            if (_layerStereo) DrawLabel(_lblStereo, _lblStereoPos, tint);
            if (_layerSpectrum) DrawLabel(_lblSpectrum, _lblSpectrumPos, tint);
            if (_layerWave) DrawLabel(_lblWave, _lblWavePos, tint);
        }

        SwapBuffers();
    }

    private void BuildSpectrumVertices()
    {
        Vector2 center = _center;

        float inner = _innerRadius;
        float maxLen = _barMaxLen;

        // tiny base so it never collapses to nothing
        float minLen = MathF.Max(6f, maxLen * 0.08f);

        // angle per bar, with small gap
        float gap = MathHelper.DegreesToRadians(SpectrumGapDegrees);

        // Нижняя точка окружности в нашей системе углов — это Pi/2 (вниз по экрану)
        float bottom = MathHelper.PiOver2;

        // Низы стартуют чуть левее низа: bottom + gap/2
        float startAngle = bottom + gap * 0.5f;

        // Дуга, по которой рисуем палочки (почти полный круг, но с разрывом снизу)
        float arc = MathHelper.TwoPi - gap;

        // чтобы последние “высокие” закончились симметрично справа от низа: bottom - gap/2
        float step = arc / (BarsCount - 1);

        for (int i = 0; i < BarsCount; i++)
        {
            float a = startAngle + i * step;
            Vector2 dir = new(MathF.Cos(a), MathF.Sin(a));
            Vector2 nrm = new(-dir.Y, dir.X);

            float mag = _barSmoothed[i]; // 0..1

            // 1) Поднимаем тихие уровни (soft-gain): быстро вытягивает малые значения
            const float gain = 3.2f;                // 2.0..5.0 (больше => больше тянет тихие)
            float lifted = 1f - MathF.Exp(-mag * gain);

            // 2) Гамма < 1 дополнительно усиливает низкие уровни
            const float gamma = 0.65f;              // 0.50..0.85 (меньше => сильнее тянет тихие)
            lifted = MathF.Pow(lifted, gamma);

            // 3) Базовая длина в доле от maxLen (чтобы “почти тишина” тоже была видна)
            float baseFrac = 0.10f;                 // 0.06..0.18
            float len = (baseFrac + (1f - baseFrac) * lifted) * maxLen;
            len = MathF.Max(len, minLen);

            // color gradient by frequency index (low->mid->high)
            float t = i / (float)(BarsCount - 1);
            Vector4 col = SpectrumGradient(t);
            // subtle alpha scaling with magnitude
            col.W = 0.25f + 0.75f * MathHelper.Clamp(mag, 0f, 1f);

            float thickness = _barThickness;
            Vector2 p0 = center + dir * inner;
            Vector2 p1 = center + dir * (inner + len);

            Vector2 o = nrm * (thickness * 0.5f);

            // quad => 2 triangles (6 verts)
            int v = i * 6;
            _spectrumVerts[v + 0] = new Vertex(p0 - o, col);
            _spectrumVerts[v + 1] = new Vertex(p1 - o, col);
            _spectrumVerts[v + 2] = new Vertex(p1 + o, col);

            _spectrumVerts[v + 3] = new Vertex(p0 - o, col);
            _spectrumVerts[v + 4] = new Vertex(p1 + o, col);
            _spectrumVerts[v + 5] = new Vertex(p0 + o, col);
        }
    }

    private void BuildWaveVertices()
    {
        float w = Size.X;

        float margin = _waveSideMargin;
        float baseY = _waveBaseY;
        float amp = _waveAmp;

        var col = new Vector4(0.92f, 0.92f, 1.0f, 0.85f);

        for (int i = 0; i < WaveSize; i++)
        {
            float x = MathHelper.Lerp(margin, w - margin, i / (float)(WaveSize - 1));
            float y = baseY + _wave[i] * amp;
            _waveVerts[i] = new Vertex(new Vector2(x, y), col);
        }
    }

    private int BuildParticleVertices()
    {
        int count = _particles.Count;
        if (count <= 0) return 0;

        int i = 0;
        foreach (var p in _particles.LiveParticles())
        {
            // point primitive: position + color (alpha already inside)
            _particleVerts[i++] = new Vertex(p.Position, p.Color);
            if (i >= _particleVerts.Length) break;
        }
        return i;
    }

    private void UploadAndDraw(Vertex[] data, int count, PrimitiveType prim, float lineWidth = 1f, float pointSize = 1f)
    {
        int bytes = count * Vertex.SizeInBytes;
        GL.BufferSubData(BufferTarget.ArrayBuffer, IntPtr.Zero, bytes, data);

        if (prim == PrimitiveType.LineStrip) GL.LineWidth(lineWidth);
        if (prim == PrimitiveType.Points) GL.PointSize(pointSize);

        GL.DrawArrays(prim, 0, count);
    }

    // Low=dark red, mid=yellow, high=green
    private static Vector4 SpectrumGradient(float t)
    {
        t = MathHelper.Clamp(t, 0f, 1f);

        Vector3 low = new(0.40f, 0.02f, 0.02f);
        Vector3 mid = new(1.00f, 0.90f, 0.10f);
        Vector3 high = new(0.10f, 1.00f, 0.25f);

        Vector3 rgb = (t < 0.5f)
            ? Vector3.Lerp(low, mid, t / 0.5f)
            : Vector3.Lerp(mid, high, (t - 0.5f) / 0.5f);

        return new Vector4(rgb, 1f);
    }
}

#endregion

#region Audio Analyzer (loopback + FFT + beats + BPM)

internal readonly record struct BeatEvent(float Strength, Vector4 Color);

internal sealed class AudioAnalyzer : IDisposable
{
    public event Action<BeatEvent>? OnBeat;

    private readonly int _barsCount;
    private readonly int _waveSize;
    private readonly int _fftSize;
    private readonly float _minFreq;
    private readonly float _maxFreq;
    private readonly int _hopSize;


    private WasapiLoopbackCapture? _cap;
    private WaveFormat? _wf;

    private readonly object _barsLock = new();
    private readonly float[] _barsA;
    private readonly float[] _barsB;
    private float[] _barsFront;
    private float[] _barsBack;

    private readonly object _waveLock = new();
    private readonly float[] _waveRing;
    private int _waveWrite;
    private readonly object _lrLock = new();
    private readonly float[] _lrRingL;
    private readonly float[] _lrRingR;
    private int _lrWrite;

    private readonly List<double> _intervals = new(16);

    private readonly float[] _fftRing;
    private int _fftWrite;
    private int _samplesSinceFft;

    private Complex[] _fft;
    private readonly float[] _window;
    private readonly int _fftM;

    private int[] _binIndex = Array.Empty<int>();
    private float[] _freqWeight = Array.Empty<float>();

    private readonly ConcurrentQueue<BeatEvent> _beatQueue = new();

    // Beat detection + BPM
    private double _lastDataTimeSec;
    private readonly Stopwatch _sw = Stopwatch.StartNew();

    private float _emaEnergy;
    private float _emaVar;
    private double _lastBeatTimeSec;
    private readonly Queue<double> _beatTimes = new();
    private float _bpmSmoothed = 120f;

    public float BpmSmoothed => _bpmSmoothed;

    public AudioAnalyzer(int barsCount, int waveSize, int fftSize, float minFreq, float maxFreq)
    {
        _barsCount = barsCount;
        _waveSize = waveSize;
        _fftSize = fftSize;
        _minFreq = minFreq;
        _maxFreq = maxFreq;

        _barsA = new float[_barsCount];
        _barsB = new float[_barsCount];
        _barsFront = _barsA;
        _barsBack = _barsB;

        _waveRing = new float[_waveSize];
        _lrRingL = new float[_waveSize];
        _lrRingR = new float[_waveSize];

        _fftRing = new float[_fftSize];
        _fft = new Complex[_fftSize];

        _fftM = (int)Math.Round(Math.Log(_fftSize, 2));
        if ((1 << _fftM) != _fftSize)
            throw new ArgumentException("fftSize must be a power of two.");

        _hopSize = Math.Max(64, _fftSize / 8); // ✅ чаще апдейты спектра/бита

        _window = new float[_fftSize];
        for (int i = 0; i < _fftSize; i++)
        {
            // Hann window
            _window[i] = 0.5f * (1f - MathF.Cos(MathHelper.TwoPi * i / (_fftSize - 1)));
        }
    }

    public void Start()
    {
        _cap = new WasapiLoopbackCapture(); // default output device loopback
        _wf = _cap.WaveFormat;

        BuildLogBins(_wf.SampleRate);

        _cap.DataAvailable += OnData;
        _cap.RecordingStopped += (_, __) => { };
        _cap.StartRecording();
    }

    public bool TryDequeueBeat(out BeatEvent beat) => _beatQueue.TryDequeue(out beat);

    public void CopyBars(float[] dest)
    {
        if (dest.Length != _barsCount) throw new ArgumentException("bars array size mismatch.");
        lock (_barsLock)
            Array.Copy(_barsFront, dest, _barsCount);

        // if no data recently -> calm to silence
        double now = _sw.Elapsed.TotalSeconds;
        if (now - _lastDataTimeSec > 0.20)
            Array.Clear(dest, 0, dest.Length);
    }

    public void CopyStereo(float[] left, float[] right)
    {
        if (left.Length != _waveSize || right.Length != _waveSize)
            throw new ArgumentException("stereo arrays size mismatch.");

        lock (_lrLock)
        {
            int idx = _lrWrite;
            for (int i = 0; i < _waveSize; i++)
            {
                left[i] = _lrRingL[idx];
                right[i] = _lrRingR[idx];
                idx++;
                if (idx >= _waveSize) idx = 0;
            }
        }
    }


    public void CopyWaveform(float[] dest)
    {
        if (dest.Length != _waveSize) throw new ArgumentException("wave array size mismatch.");
        lock (_waveLock)
        {
            // output in chronological order
            int idx = _waveWrite;
            for (int i = 0; i < _waveSize; i++)
            {
                dest[i] = _waveRing[idx];
                idx++;
                if (idx >= _waveSize) idx = 0;
            }
        }
    }

    private void OnData(object? sender, WaveInEventArgs e)
    {
        if (_cap is null || _wf is null) return;

        _lastDataTimeSec = _sw.Elapsed.TotalSeconds;

        int channels = _wf.Channels;

        if (_wf.Encoding == WaveFormatEncoding.IeeeFloat && _wf.BitsPerSample == 32)
        {
            var wb = new WaveBuffer(e.Buffer);
            int floatCount = e.BytesRecorded / 4;
            int frames = floatCount / channels;

            // ✅ один проход + один lock на wave/stereo кольца
            lock (_waveLock)
                lock (_lrLock)
                {
                    int wIdx = _waveWrite;
                    int lrIdx = _lrWrite;

                    int i = 0;
                    for (int f = 0; f < frames; f++, i += channels)
                    {
                        float L = wb.FloatBuffer[i + 0];
                        float R = (channels > 1) ? wb.FloatBuffer[i + 1] : L;

                        // если каналов больше 2 — мягко подмешаем остальные в середину
                        if (channels > 2)
                        {
                            float extra = 0f;
                            for (int c = 2; c < channels; c++) extra += wb.FloatBuffer[i + c];
                            extra /= (channels - 2);

                            float mid = (L + R) * 0.5f;
                            mid = (mid + extra) * 0.5f;

                            L = (L + mid) * 0.5f;
                            R = (R + mid) * 0.5f;
                        }

                        float mono = (L + R) * 0.5f;

                        // waveform ring
                        _waveRing[wIdx] = mono;
                        if (++wIdx >= _waveSize) wIdx = 0;

                        // stereo ring
                        _lrRingL[lrIdx] = L;
                        _lrRingR[lrIdx] = R;
                        if (++lrIdx >= _waveSize) lrIdx = 0;

                        // fft ring (без lock — только аудиопоток трогает)
                        _fftRing[_fftWrite] = mono;
                        if (++_fftWrite >= _fftSize) _fftWrite = 0;

                        if (++_samplesSinceFft >= _hopSize)
                        {
                            _samplesSinceFft = 0;
                            ComputeSpectrumAndBeat(_hopSize);
                        }
                    }

                    _waveWrite = wIdx;
                    _lrWrite = lrIdx;
                }
        }
        else if (_wf.BitsPerSample == 16)
        {
            int bytesPerSample = 2;
            int frameBytes = bytesPerSample * channels;
            int frames = e.BytesRecorded / frameBytes;

            int offset = 0;

            lock (_waveLock)
                lock (_lrLock)
                {
                    int wIdx = _waveWrite;
                    int lrIdx = _lrWrite;

                    for (int f = 0; f < frames; f++)
                    {
                        short sL = BitConverter.ToInt16(e.Buffer, offset); offset += 2;
                        short sR = (channels > 1) ? BitConverter.ToInt16(e.Buffer, offset) : sL;
                        if (channels > 1) offset += 2;

                        // пропускаем остальные каналы
                        for (int c = 2; c < channels; c++) offset += 2;

                        float L = sL / 32768f;
                        float R = sR / 32768f;
                        float mono = (L + R) * 0.5f;

                        _waveRing[wIdx] = mono;
                        if (++wIdx >= _waveSize) wIdx = 0;

                        _lrRingL[lrIdx] = L;
                        _lrRingR[lrIdx] = R;
                        if (++lrIdx >= _waveSize) lrIdx = 0;

                        _fftRing[_fftWrite] = mono;
                        if (++_fftWrite >= _fftSize) _fftWrite = 0;

                        if (++_samplesSinceFft >= _hopSize)
                        {
                            _samplesSinceFft = 0;
                            ComputeSpectrumAndBeat(_hopSize);
                        }
                    }

                    _waveWrite = wIdx;
                    _lrWrite = lrIdx;
                }
        }
        // else: unsupported format ignore
    }

    private void PushSample(float mono, float left, float right)
    {
        // waveform ring (mono)
        lock (_waveLock)
        {
            _waveRing[_waveWrite] = mono;
            _waveWrite++;
            if (_waveWrite >= _waveSize) _waveWrite = 0;
        }

        // stereo ring
        lock (_lrLock)
        {
            _lrRingL[_lrWrite] = left;
            _lrRingR[_lrWrite] = right;
            _lrWrite++;
            if (_lrWrite >= _waveSize) _lrWrite = 0;
        }

        // fft ring (mono)
        _fftRing[_fftWrite] = mono;
        _fftWrite++;
        if (_fftWrite >= _fftSize) _fftWrite = 0;

        _samplesSinceFft++;

        int hop = _fftSize / 2;
        if (_samplesSinceFft >= hop)
        {
            _samplesSinceFft = 0;
            ComputeSpectrumAndBeat(hop);
        }
    }

    private void ComputeSpectrumAndBeat(int hop)
    {
        if (_wf is null) return;

        // build contiguous FFT input from ring (oldest starts at _fftWrite)
        int start = _fftWrite;
        for (int i = 0; i < _fftSize; i++)
        {
            float x = _fftRing[start] * _window[i];
            _fft[i].X = x;
            _fft[i].Y = 0f;

            start++;
            if (start >= _fftSize) start = 0;
        }

        FastFourierTransform.FFT(true, _fftM, _fft);

        // magnitudes for bins 0..N/2
        int nyq = _fftSize / 2;

        // Build bars (log-frequency bins)
        float domW = 0f;
        float domT = 0f;

        for (int b = 0; b < _barsCount; b++)
        {
            int k = _binIndex[b];

            // берём небольшое усреднение по соседям, чтобы не дрожало
            float mag = 0f;
            for (int kk = k - 1; kk <= k + 1; kk++)
            {
                float re = _fft[kk].X;
                float im = _fft[kk].Y;
                mag += MathF.Sqrt(re * re + im * im);
            }
            mag /= 3f;

            // ✅ частотная компенсация: убираем “перебас”, оживляем верха
            mag *= _freqWeight[b];
            // компрессия в 0..1
            float db = 20f * MathF.Log10(mag + 1e-8f);
            float norm = (db + 60f) / 60f;
            norm = MathHelper.Clamp(norm, 0f, 1f);

            _barsBack[b] = norm;

            float t = b / (float)(_barsCount - 1);
            domW += norm;
            domT += norm * t;
        }


        // swap bars buffers (tiny lock)
        lock (_barsLock)
        {
            (_barsFront, _barsBack) = (_barsBack, _barsFront);
        }

        // Dominant spectrum color (weighted)
        float dom = domT / (domW + 1e-6f);
        Vector4 domColor = VisualizerWindowSpectrumGradient(dom);
        domColor.W = 1f;

        // Beat energy from low band (approx 30..160 Hz)
        float lowEnergy = LowBandEnergy(_wf.SampleRate, 30f, 160f, nyq);

        // Adaptive threshold using EMA + variance
        float dt = (float)(hop / (double)_wf.SampleRate);
        float aE = 1f - MathF.Exp(-dt / 0.35f); // energy smoothing
        float aV = 1f - MathF.Exp(-dt / 0.55f); // variance smoothing

        float diff = lowEnergy - _emaEnergy;
        _emaEnergy += diff * aE;
        _emaVar += ((diff * diff) - _emaVar) * aV;

        float sigma = MathF.Sqrt(MathF.Max(_emaVar, 1e-9f));
        float thr = _emaEnergy + 2.2f * sigma;

        double now = _sw.Elapsed.TotalSeconds;
        double minInterval = 0.18; // ~333 bpm max

        if (lowEnergy > thr && (now - _lastBeatTimeSec) > minInterval)
        {
            _lastBeatTimeSec = now;

            float strength = (lowEnergy - thr) / (thr + 1e-6f);
            strength = MathHelper.Clamp(strength, 0f, 1f);

            var beat = new BeatEvent(strength, domColor);
            _beatQueue.Enqueue(beat);
            OnBeat?.Invoke(beat);

            UpdateBpm(now);
        }

        // Smooth BPM even without beats
        _bpmSmoothed = MathHelper.Lerp(_bpmSmoothed, _bpmSmoothed, 0f);
    }

    private float LowBandEnergy(int sampleRate, float f0, float f1, int nyq)
    {
        int b0 = (int)(f0 * _fftSize / sampleRate);
        int b1 = (int)(f1 * _fftSize / sampleRate);
        b0 = Math.Clamp(b0, 1, nyq - 1);
        b1 = Math.Clamp(b1, b0 + 1, nyq);

        float sum = 0f;
        for (int k = b0; k < b1; k++)
        {
            float re = _fft[k].X;
            float im = _fft[k].Y;
            float mag = MathF.Sqrt(re * re + im * im);
            sum += mag;
        }
        return sum / (b1 - b0);
    }

    private void UpdateBpm(double now)
    {
        _beatTimes.Enqueue(now);
        while (_beatTimes.Count > 12)
            _beatTimes.Dequeue();

        if (_beatTimes.Count < 4) return;

        _intervals.Clear();

        double prev = double.NaN;
        foreach (var t in _beatTimes)
        {
            if (!double.IsNaN(prev))
            {
                double d = t - prev;
                if (d > 0.18 && d < 1.20)
                    _intervals.Add(d);
            }
            prev = t;
        }

        if (_intervals.Count < 3) return;

        _intervals.Sort();
        double median = _intervals[_intervals.Count / 2];
        float bpm = (float)(60.0 / median);

        _bpmSmoothed = MathHelper.Lerp(_bpmSmoothed, bpm, 0.22f);
    }

    private void BuildLogBins(int sampleRate)
    {
        int nyq = _fftSize / 2;
        _binIndex = new int[_barsCount];

        float ratio = _maxFreq / _minFreq;

        // 1) хотим лог-распределённые ЦЕНТРЫ полос (в бинах)
        for (int i = 0; i < _barsCount; i++)
        {
            float t = i / (float)(_barsCount - 1);
            float f = _minFreq * MathF.Pow(ratio, t);

            int k = (int)MathF.Round(f * _fftSize / sampleRate);
            k = Math.Clamp(k, 1, nyq - 2); // -2 чтобы можно было брать k+1
            _binIndex[i] = k;
        }

        // 2) убираем дубликаты: делаем строго возрастающую последовательность
        for (int i = 1; i < _barsCount; i++)
        {
            if (_binIndex[i] <= _binIndex[i - 1])
                _binIndex[i] = _binIndex[i - 1] + 1;
        }

        // 3) если "выперли" за nyq — подтягиваем назад, сохраняя уникальность
        if (_binIndex[^1] > nyq - 2)
        {
            _binIndex[^1] = nyq - 2;
            for (int i = _barsCount - 2; i >= 0; i--)
            {
                if (_binIndex[i] >= _binIndex[i + 1])
                    _binIndex[i] = _binIndex[i + 1] - 1;
            }

            // и на всякий случай ограничим снизу
            for (int i = 0; i < _barsCount; i++)
                _binIndex[i] = Math.Max(_binIndex[i], 1);
        }
        _freqWeight = new float[_barsCount];

        // Компенсация наклона спектра: low меньше, high больше
        // Параметры можно подстроить: exponent 0.25..0.55
        const float exponent = 0.40f;

        for (int i = 0; i < _barsCount; i++)
        {
            float freq = _binIndex[i] * sampleRate / (float)_fftSize;

            // база относительно 1 кГц
            float w = MathF.Pow(freq / 1000f, exponent);

            // ограничим, чтобы не улетало в крайностях
            w = Math.Clamp(w, 0.35f, 2.8f);

            _freqWeight[i] = w;
        }
    }


    private static Vector4 VisualizerWindowSpectrumGradient(float t)
    {
        t = MathHelper.Clamp(t, 0f, 1f);

        Vector3 low = new(0.40f, 0.02f, 0.02f);
        Vector3 mid = new(1.00f, 0.90f, 0.10f);
        Vector3 high = new(0.10f, 1.00f, 0.25f);

        Vector3 rgb = (t < 0.5f)
            ? Vector3.Lerp(low, mid, t / 0.5f)
            : Vector3.Lerp(mid, high, (t - 0.5f) / 0.5f);

        return new Vector4(rgb, 1f);
    }

    public void Dispose()
    {
        if (_cap is not null)
        {
            try { _cap.StopRecording(); } catch { }
            _cap.DataAvailable -= OnData;
            _cap.Dispose();
            _cap = null;
        }
    }
}

#endregion

#region Particles

internal readonly record struct ParticleSnapshot(Vector2 Position, Vector4 Color);

internal sealed class ParticleSystem
{
    public int MaxParticles { get; }
    private readonly List<Particle> _particles;
    private readonly Random _rng = new();

    public int Count => _particles.Count;

    public ParticleSystem(int maxParticles)
    {
        MaxParticles = maxParticles;
        _particles = new List<Particle>(maxParticles);
    }

    public void Clear() => _particles.Clear();

    public void SpawnBurst(Vector2 center, int count, float strength, Vector4 baseColor, Vector2i windowSize)
    {
        // intensity -> faster + more particles
        float minSpeed = 700f;
        float maxSpeed = 1700f;
        float speed = MathHelper.Lerp(minSpeed, maxSpeed, strength);

        // tint baseColor a bit towards white on strong hits
        Vector4 c = baseColor;
        c.X = MathHelper.Lerp(c.X, 1f, 0.15f * strength);
        c.Y = MathHelper.Lerp(c.Y, 1f, 0.15f * strength);
        c.Z = MathHelper.Lerp(c.Z, 1f, 0.15f * strength);

        for (int i = 0; i < count; i++)
        {
            if (_particles.Count >= MaxParticles) break;

            float a = (float)(_rng.NextDouble() * MathHelper.TwoPi);
            float sp = speed * (0.65f + 0.75f * (float)_rng.NextDouble());

            Vector2 vel = new(MathF.Cos(a), MathF.Sin(a));
            vel *= sp;

            float life = 2.0f + 1.0f * strength; // 2..3 секунды
            float size = 2.0f + 2.0f * strength;

            // alpha stronger when hit stronger
            Vector4 col = new(c.X, c.Y, c.Z, 0.75f + 0.25f * strength);

            _particles.Add(new Particle(center, vel, 0f, life, size, col));
        }
    }

    public void Update(float dt, Vector2i windowSize)
    {
        // update + cull
        for (int i = _particles.Count - 1; i >= 0; i--)
        {
            var p = _particles[i];

            p.Age += dt;
            if (p.Age >= p.Life)
            {
                _particles.RemoveAt(i);
                continue;
            }

            // friction
            p.Vel *= MathF.Exp(-dt * 1.7f);
            p.Pos += p.Vel * dt;

            // fade (fast start, smooth decay to 0)
            // fade: плавное затухание к 0 за весь Life (без “ускорения” из-за перемножения)
            float t = p.Age / p.Life;
            t = MathHelper.Clamp(t, 0f, 1f);

            // smoothstep 0..1: s = t*t*(3-2t), затем инвертируем (1 - s)
            float s = t * t * (3f - 2f * t);
            float fade = 1f - s;

            // считаем альфу от исходной, а не умножаем каждый кадр
            p.Color.W = p.BaseAlpha * fade;

            // soft bounds kill (optional)
            if (p.Pos.X < -50 || p.Pos.X > windowSize.X + 50 || p.Pos.Y < -50 || p.Pos.Y > windowSize.Y + 50)
            {
                p.Color.W *= 0.90f;
            }

            _particles[i] = p;
        }
    }

    public IEnumerable<ParticleSnapshot> LiveParticles()
    {
        foreach (var p in _particles)
        {
            var c = p.Color;
            // keep alpha in a nice range
            c.W = MathHelper.Clamp(c.W, 0f, 1f);
            yield return new ParticleSnapshot(p.Pos, c);
        }
    }

    private struct Particle
    {
        public Vector2 Pos;
        public Vector2 Vel;
        public float Age;
        public float Life;
        public float Size;

        public Vector4 Color;
        public float BaseAlpha; // исходная альфа (не меняется)

        public Particle(Vector2 pos, Vector2 vel, float age, float life, float size, Vector4 color)
        {
            Pos = pos;
            Vel = vel;
            Age = age;
            Life = life;
            Size = size;

            Color = color;
            BaseAlpha = color.W;
        }
    }
}

#endregion

#region GL Helpers

[StructLayout(LayoutKind.Sequential)]
internal struct Vertex
{
    public Vector2 Pos;
    public Vector4 Color;

    public Vertex(Vector2 pos, Vector4 color)
    {
        Pos = pos;
        Color = color;
    }

    public const int SizeInBytes = (2 + 4) * 4;
}

internal sealed class ShaderProgram : IDisposable
{
    public int Handle { get; }

    public ShaderProgram(string vertSource, string fragSource)
    {
        int vs = Compile(ShaderType.VertexShader, vertSource);
        int fs = Compile(ShaderType.FragmentShader, fragSource);

        Handle = GL.CreateProgram();
        GL.AttachShader(Handle, vs);
        GL.AttachShader(Handle, fs);
        GL.LinkProgram(Handle);

        GL.GetProgram(Handle, GetProgramParameterName.LinkStatus, out int ok);
        if (ok == 0)
        {
            string log = GL.GetProgramInfoLog(Handle);
            throw new Exception("Shader link failed:\n" + log);
        }

        GL.DetachShader(Handle, vs);
        GL.DetachShader(Handle, fs);
        GL.DeleteShader(vs);
        GL.DeleteShader(fs);
    }

    public void SetUniform(string name, Vector4 v)
    {
        int loc = GL.GetUniformLocation(Handle, name);
        if (loc >= 0) GL.Uniform4(loc, v);
    }

    public void SetUniform(string name, int v)
    {
        int loc = GL.GetUniformLocation(Handle, name);
        if (loc >= 0) GL.Uniform1(loc, v);
    }


    public void Use() => GL.UseProgram(Handle);

    public void SetUniform(string name, float v)
    {
        int loc = GL.GetUniformLocation(Handle, name);
        if (loc >= 0) GL.Uniform1(loc, v);
    }

    public void SetUniform(string name, Vector2 v)
    {
        int loc = GL.GetUniformLocation(Handle, name);
        if (loc >= 0) GL.Uniform2(loc, v);
    }

    private static int Compile(ShaderType type, string src)
    {
        int id = GL.CreateShader(type);
        GL.ShaderSource(id, src);
        GL.CompileShader(id);
        GL.GetShader(id, ShaderParameter.CompileStatus, out int ok);
        if (ok == 0)
        {
            string log = GL.GetShaderInfoLog(id);
            throw new Exception($"{type} compile failed:\n{log}");
        }
        return id;
    }

    public void Dispose()
    {
        if (Handle != 0) GL.DeleteProgram(Handle);
    }
}

internal static class Shaders
{
    public const string ColorVert = @"
#version 330 core
layout(location=0) in vec2 aPos;
layout(location=1) in vec4 aColor;
out vec4 vColor;
uniform vec2 uResolution;
void main()
{
    vec2 ndc = (aPos / uResolution) * 2.0 - 1.0;
    ndc.y = -ndc.y;
    gl_Position = vec4(ndc, 0.0, 1.0);
    vColor = aColor;
}";

    public const string TextVert = @"
#version 330 core
layout(location=0) in vec2 aPos;
layout(location=1) in vec2 aUV;
out vec2 vUV;
uniform vec2 uResolution;
void main()
{
    vec2 ndc = (aPos / uResolution) * 2.0 - 1.0;
    ndc.y = -ndc.y;
    gl_Position = vec4(ndc, 0.0, 1.0);
    vUV = aUV;
}";

    public const string TextFrag = @"
#version 330 core
in vec2 vUV;
out vec4 FragColor;
uniform sampler2D uTex;
uniform vec4 uTint;
void main()
{
    vec4 t = texture(uTex, vUV);
    FragColor = vec4(uTint.rgb, uTint.a * t.a);
}";


    public const string ColorFrag = @"
#version 330 core
in vec4 vColor;
out vec4 FragColor;
void main()
{
    FragColor = vColor;
}";

    public const string BgVert = @"
#version 330 core
layout(location=0) in vec2 aPos;
out vec2 vUV;
void main()
{
    vUV = aPos * 0.5 + 0.5;
    gl_Position = vec4(aPos, 0.0, 1.0);
}";

    // subtle moving fog/noise, speed depends on BPM
    public const string BgFrag = @"
#version 330 core
in vec2 vUV;
out vec4 FragColor;

uniform float uTime;
uniform float uSpeed;
uniform float uBeatGlow;
uniform vec4  uBeatTint;

float hash(vec2 p) { return fract(sin(dot(p, vec2(127.1, 311.7))) * 43758.5453); }

float noise(vec2 p)
{
    vec2 i = floor(p);
    vec2 f = fract(p);
    float a = hash(i);
    float b = hash(i + vec2(1.0, 0.0));
    float c = hash(i + vec2(0.0, 1.0));
    float d = hash(i + vec2(1.0, 1.0));
    vec2 u = f*f*(3.0 - 2.0*f);
    return mix(a, b, u.x) + (c - a) * u.y * (1.0 - u.x) + (d - b) * u.x * u.y;
}

float fbm(vec2 p)
{
    float v = 0.0;
    float a = 0.55;
    for (int i = 0; i < 5; i++)
    {
        v += a * noise(p);
        p *= 2.0;
        a *= 0.5;
    }
    return v;
}

void main()
{
    vec2 uv = vUV;

    vec2 p = uv * 3.4;
    float t = uTime * 0.14 * uSpeed;
    p += vec2(t, t * 0.7);

    vec2 c = uv - vec2(0.5, 0.5);
    float r = length(c);
    float centerMask = 1.0 - smoothstep(0.0, 0.85, r);

    p += c * (0.55 * uBeatGlow * centerMask);
    p += vec2(sin(uTime * 6.0), cos(uTime * 5.0)) * (0.10 * uBeatGlow);

    float n = fbm(p);

    float lo = 0.25 - 0.08 * uBeatGlow;
    float hi = 0.88 - 0.03 * uBeatGlow;
    float fog = smoothstep(lo, hi, n);

    vec3 base = vec3(0.0);

    vec3 tintBase = vec3(0.085, 0.035, 0.12);
    vec3 beatTint = uBeatTint.rgb * 0.22;
    vec3 tint = tintBase + beatTint * uBeatGlow;

    float glow = 0.32 * uBeatGlow;
    vec3 col = base + tint * (0.55 + glow) * fog;

    vec2 q = uv * 2.0 - 1.0;
    float vig = smoothstep(1.3, 0.4, dot(q, q));
    col *= (0.78 + 0.22 * vig);

    FragColor = vec4(col, 1.0);
}";


}

#endregion
