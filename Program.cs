using NAudio.Dsp;
using NAudio.Wave;
using OpenTK.Compute.OpenCL;
using OpenTK.Graphics.OpenGL4;
using OpenTK.Mathematics;
using OpenTK.Windowing.Common;
using OpenTK.Windowing.Desktop;
using NAudio.Wave.SampleProviders;
using NAudio.CoreAudioApi;
using OpenTK.Windowing.GraphicsLibraryFramework;
using System;
using System.Collections.Concurrent;
using System.Diagnostics;
using System.Drawing;
using System.Drawing.Imaging;
using System.Drawing.Text;
using System.Runtime.InteropServices;
using System;
using System.IO;
using System.Buffers;
using System.IO;
using System.Text.Json;
using System.Threading;
using System.Linq;
using System.Threading.Tasks;
using Forms = System.Windows.Forms;
using TkKeys = OpenTK.Windowing.GraphicsLibraryFramework.Keys;
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
                Title = "...",
                WindowBorder = WindowBorder.Resizable,
                StartVisible = true,
                APIVersion = new Version(3, 3),
                Profile = ContextProfile.Core,
                Flags = ContextFlags.ForwardCompatible,

                NumberOfSamples = 8, // 4 или 8 (8 красивее, но тяжелее)
            };


            using var window = new VisualizerWindow(game, native);
            AppDomain.CurrentDomain.UnhandledException += (_, ev) =>
            {
                MessageBoxW(IntPtr.Zero, ev.ExceptionObject?.ToString() ?? "Unknown", "UnhandledException", 0);
            };
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
    private const int WaveSize = 4096;
    // где-нибудь рядом с константами (по желанию)
    private const int MaxWaveDrawPoints = 8192; // можно 8192/16384/32768, но <= 100000 (см. VBO)
    private const int MinWaveDrawPoints = 2048;
    // oscilloscope samples
    private const int FftSize = 2048;           // 1024 or 2048 (latency vs detail)
    private const float MinFreq = 20f;
    private const float MaxFreq = 20000f;
    private const float SpectrumGapDegrees = 40f;
    private int _mp3AutoNextFlag = 0;

    private int _waveDrawPoints = WaveSize; // сколько точек реально рисуем по экрану

    private float _wavePad;        // внутренний отступ зоны волны
    private float _waveYMin;       // верхняя граница (в пикселях)
    private float _waveYMax;       // нижняя граница (в пикселях)
    private float _waveHalfTh;     // half thickness ленты (в пикселях)

    private bool _hideUi = false;

    // ===== MP3 menu overlay =====
    private bool _mp3MenuOpen = false;
    private int _mp3MenuIndex = 0;
    private bool _mp3MenuDirty = true;

    private TextLabel _mp3MenuTitle;
    private TextLabel _mp3MenuHelp;
    private TextLabel _mp3MenuEmpty;

    private TextLabel[] _mp3MenuItems = Array.Empty<TextLabel>();
    private Vector2[] _mp3MenuItemPos = Array.Empty<Vector2>();

    private int _mp3MenuFontPx = -1;

    private float _mp3MenuX, _mp3MenuY, _mp3MenuW, _mp3MenuH;
    private float _mp3MenuPad;
    private float _mp3MenuRowH;

    private Vertex[] _mp3MenuPanelVerts = new Vertex[6];
    private Vertex[] _mp3MenuSelectVerts = new Vertex[6];

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
    private readonly Mp3Library _mp3Lib = Mp3Library.Load();
    private LoopbackAudioEngine? _loopback;
    private Mp3AudioEngine? _mp3;
    private AudioMode _audioMode = AudioMode.Loopback;

    private IAudioEngine Audio
    {
        get
        {
            if (_audioMode == AudioMode.Mp3 && _mp3 is not null) return _mp3;
            return _loopback ?? throw new InvalidOperationException("Loopback engine not initialized.");
        }
    }

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

    private float _waveDisplayGain = 1f;

    private double _waveRefreshAcc = 0.0;

    // Particles
    private readonly ParticleSystem _particles = new(maxParticles: 4000);

    // GL: shaders
    private ShaderProgram? _shaderColor;
    private ShaderProgram? _shaderBackground;

    // GL: VAO/VBO for dynamic geometry
    private int _vaoDynamic, _vboDynamic;
    private int _vaoBg, _vboBg;

    private void UpdateWaveDisplayGain(float frameDt)
    {
        // max амплитуда текущего буфера
        float maxAbs = 1e-6f;
        for (int i = 0; i < _wave.Length; i++)
        {
            float a = MathF.Abs(_wave[i]);
            if (a > maxAbs) maxAbs = a;
        }

        // хотим запас, чтобы Catmull/пики не упирались в yMin/yMax
        float targetGain = MathF.Min(1f, 0.92f / maxAbs);

        // быстро вниз, медленно вверх
        float down = 1f - MathF.Exp(-frameDt / 0.040f);
        float up = 1f - MathF.Exp(-frameDt / 0.220f);

        float lerpGain = (targetGain < _waveDisplayGain) ? down : up;
        _waveDisplayGain += (targetGain - _waveDisplayGain) * lerpGain;
    }


    // CPU vertex buffers (reused)
    // CPU vertex buffers (reused)
    private Vertex[] _spectrumVerts = Array.Empty<Vertex>();
    private Vertex[] _waveVerts = Array.Empty<Vertex>();
    // Wave (ribbon + outline)
    private Vertex[] _waveStripVerts = Array.Empty<Vertex>(); // TriangleStrip: _waveDrawPoints * 2
    private Vertex[] _waveLineVerts = Array.Empty<Vertex>(); // LineStrip:     _waveDrawPoints
    private Vector2[] _wavePts = Array.Empty<Vector2>(); // temp points:  _waveDrawPoints
    private Vertex[] _particleVerts = Array.Empty<Vertex>();
    private Vertex[] _circleVerts = Array.Empty<Vertex>();


    private double _timeSec;
    private double _lastBeatFlash; // for subtle bg emphasis

    private readonly float[] _tmpL = new float[WaveSize];
    private readonly float[] _tmpR = new float[WaveSize];

    public VisualizerWindow(GameWindowSettings gameWindowSettings, NativeWindowSettings nativeWindowSettings)
        : base(gameWindowSettings, nativeWindowSettings) { }

    private static string Shorten(string s, int maxChars)
    {
        if (string.IsNullOrEmpty(s) || s.Length <= maxChars) return s;
        return s.Substring(0, Math.Max(0, maxChars - 1)) + "…";
    }

    private void BuildWaveRibbonVertices()
    {
        int cols = _waveDrawPoints;
        if (cols < 2) return;

        float w = Size.X;
        float margin = _waveSideMargin;
        float baseY = _waveBaseY;
        float amp = _waveAmp;

        float halfThickness = _waveHalfTh;

        var colFill = new Vector4(0.92f, 0.92f, 1.00f, 0.18f);
        var colLine = new Vector4(0.92f, 0.92f, 1.00f, 0.88f);

        float yMin = _waveYMin + halfThickness;
        float yMax = _waveYMax - halfThickness;

        for (int i = 0; i < cols; i++)
        {
            float u = i / (float)(cols - 1);
            float x = MathHelper.Lerp(margin, w - margin, u);

            float src = u * (WaveSize - 1);
            float s = SampleWaveCatmull(src) * _waveDisplayGain;

            // лучше не давать овершуту Catmull улетать выше 1 по визуалу
            s = MathHelper.Clamp(s, -1f, 1f);

            // альтернативно (ещё мягче, без "плоских" пиков):
            // s = MathF.Tanh(s * 1.35f);


            float y = baseY + s * amp;

            // КЛЮЧЕВО: не даём ленте упереться в край окна
            y = Math.Clamp(y, yMin, yMax);

            _wavePts[i] = new Vector2(x, y);
            _waveLineVerts[i] = new Vertex(_wavePts[i], colLine);
        }

        for (int i = 0; i < cols; i++)
        {
            Vector2 p = _wavePts[i];
            Vector2 p0 = _wavePts[Math.Max(i - 1, 0)];
            Vector2 p1 = _wavePts[Math.Min(i + 1, cols - 1)];

            Vector2 t = p1 - p0;
            float len = t.Length;
            if (len < 1e-5f) t = new Vector2(1, 0);
            else t /= len;

            Vector2 n = new Vector2(-t.Y, t.X);
            Vector2 o = n * halfThickness;

            int v = i * 2;
            _waveStripVerts[v + 0] = new Vertex(p - o, colFill);
            _waveStripVerts[v + 1] = new Vertex(p + o, colFill);
        }
    }

    private static void SmoothWaveForDisplay(float[] x)
    {
        // 0.25..0.55: меньше = сильнее сглаживание
        const float a = 0.15f;

        // forward
        float y = x[0];
        for (int i = 1; i < x.Length; i++)
        {
            y += (x[i] - y) * a;
            x[i] = y;
        }

        // backward (убирает фазовый сдвиг → “верх/низ не съезжают”)
        y = x[^1];
        for (int i = x.Length - 2; i >= 0; i--)
        {
            y += (x[i] - y) * a;
            x[i] = y;
        }
    }


    private void MarkMp3MenuDirty()
    {
        _mp3MenuDirty = true;
    }

    private void OnMp3TrackEnded()
    {
        Interlocked.Exchange(ref _mp3AutoNextFlag, 1);
    }

    private void PlayNextMp3_Autoplay()
    {
        if (_mp3Lib.Tracks.Count == 0) return;

        _mp3Lib.Next();                 // next (последний -> первый уже сделано внутри)
        _mp3Lib.LastMode = AudioMode.Mp3;
        _mp3Lib.Save();

        _mp3 ??= new Mp3AudioEngine(BarsCount, WaveSize, FftSize, MinFreq, MaxFreq);
        _mp3.LoadTrack(_mp3Lib.Current!, autoPlay: true);

        _vizEnabled = true;
        _particles.Clear();

        _mp3MenuIndex = _mp3Lib.SelectedIndex;
        MarkMp3MenuDirty();             // если меню открыто — выделение обновится
        UpdateWindowTitle();
    }

    private void OpenMp3Menu()
    {
        _mp3MenuOpen = true;
        _mp3MenuIndex = _mp3Lib.SelectedIndex;
        MarkMp3MenuDirty();
        UpdateWindowTitle();
    }

    private void CloseMp3Menu()
    {
        _mp3MenuOpen = false;
        UpdateWindowTitle();
    }

    // ===== layout + textures =====
    private void LayoutMp3Menu()
    {
        float w = Size.X;
        float h = Size.Y;

        _mp3MenuW = MathF.Min(w * 0.78f, 920f);
        _mp3MenuH = MathF.Min(h * 0.74f, 720f);

        _mp3MenuX = (w - _mp3MenuW) * 0.5f;
        _mp3MenuY = (h - _mp3MenuH) * 0.5f;

        _mp3MenuPad = MathF.Max(14f, _mp3MenuW * 0.03f);
    }

    private void DestroyTextLabel(ref TextLabel lbl)
    {
        if (lbl.Tex != 0) GL.DeleteTexture(lbl.Tex);
        lbl = default;
    }

    private void DestroyMenuTextures()
    {
        DestroyTextLabel(ref _mp3MenuTitle);
        DestroyTextLabel(ref _mp3MenuHelp);
        DestroyTextLabel(ref _mp3MenuEmpty);

        if (_mp3MenuItems.Length > 0)
        {
            for (int i = 0; i < _mp3MenuItems.Length; i++)
            {
                var t = _mp3MenuItems[i];
                if (t.Tex != 0) GL.DeleteTexture(t.Tex);
            }
        }

        _mp3MenuItems = Array.Empty<TextLabel>();
        _mp3MenuItemPos = Array.Empty<Vector2>();
    }

    private void EnsureMp3MenuResources()
    {
        if (!_mp3MenuOpen) return;

        // авто-лейаут (на всякий)
        if (_mp3MenuW <= 0 || _mp3MenuH <= 0) LayoutMp3Menu();

        int fontPx = (int)Math.Clamp(Size.Y * 0.024f, 15f, 22f);
        if (!_mp3MenuDirty && fontPx == _mp3MenuFontPx) return;

        _mp3MenuDirty = false;
        _mp3MenuFontPx = fontPx;

        // перестраиваем текстуры
        DestroyMenuTextures();

        _mp3MenuTitle = new TextLabel { Text = "MP3 LIBRARY" };
        _mp3MenuHelp = new TextLabel { Text = "↑/↓ - выбор   • Enter - выбрать   • Del - удалить   • Esc закрыть   • O - импорт" };
        _mp3MenuEmpty = new TextLabel { Text = "Нет загруженных MP3. Нажмите O, чтобы импортировать песни." };

        _mp3MenuTitle.Tex = CreateTextTexture(_mp3MenuTitle.Text, (int)(fontPx * 1.15f), out _mp3MenuTitle.W, out _mp3MenuTitle.H);
        _mp3MenuHelp.Tex = CreateTextTexture(_mp3MenuHelp.Text, fontPx, out _mp3MenuHelp.W, out _mp3MenuHelp.H);
        _mp3MenuEmpty.Tex = CreateTextTexture(_mp3MenuEmpty.Text, fontPx, out _mp3MenuEmpty.W, out _mp3MenuEmpty.H);

        int n = _mp3Lib.Tracks.Count;
        _mp3MenuItems = (n > 0) ? new TextLabel[n] : Array.Empty<TextLabel>();
        _mp3MenuItemPos = (n > 0) ? new Vector2[n] : Array.Empty<Vector2>();

        // строки списка
        // подберем примерную ширину в символах (грубо)
        int maxChars = (int)Math.Clamp(_mp3MenuW / (fontPx * 0.55f), 24, 80);

        for (int i = 0; i < n; i++)
        {
            string path = _mp3Lib.Tracks[i];
            string name = Path.GetFileNameWithoutExtension(path);

            // номер + имя
            string line = $"{i + 1:00}. {Shorten(name, maxChars)}";

            _mp3MenuItems[i] = new TextLabel { Text = line };
            _mp3MenuItems[i].Tex = CreateTextTexture(line, fontPx, out _mp3MenuItems[i].W, out _mp3MenuItems[i].H);
        }

        // высота строки
        _mp3MenuRowH = (n > 0) ? (_mp3MenuItems[0].H + 6f) : (fontPx + 12f);

        // позиции
        float x = _mp3MenuX + _mp3MenuPad;
        float y = _mp3MenuY + _mp3MenuPad;

        // title
        float titleY = y;
        float listTop = titleY + _mp3MenuTitle.H + 12f;

        // help снизу
        float helpY = _mp3MenuY + _mp3MenuH - _mp3MenuPad - _mp3MenuHelp.H;

        // список помещаем между listTop и helpY
        float listBottom = helpY - 14f;

        // если список не помещается — уменьшим шаг (мягко)
        int rowsFit = (int)MathF.Floor((listBottom - listTop) / _mp3MenuRowH);
        if (rowsFit < 4) rowsFit = 4;

        // центруем окно списка по выбранному
        if (n > 0)
        {
            _mp3MenuIndex = Math.Clamp(_mp3MenuIndex, 0, n - 1);
            int first = Math.Clamp(_mp3MenuIndex - rowsFit / 2, 0, Math.Max(0, n - rowsFit));
            int last = Math.Min(n, first + rowsFit);

            // запишем позиции только для видимых, остальные пометим как "за экраном"
            for (int i = 0; i < n; i++) _mp3MenuItemPos[i] = new Vector2(-9999, -9999);

            float yy = listTop;
            for (int i = first; i < last; i++)
            {
                _mp3MenuItemPos[i] = new Vector2(x, yy);
                yy += _mp3MenuRowH;
            }
        }
    }

    private void BuildRectVerts(float x, float y, float w, float h, Vector4 col, Vertex[] dst)
    {
        // 2 triangles
        dst[0] = new Vertex(new Vector2(x, y), col);
        dst[1] = new Vertex(new Vector2(x + w, y), col);
        dst[2] = new Vertex(new Vector2(x + w, y + h), col);
        dst[3] = new Vertex(new Vector2(x, y), col);
        dst[4] = new Vertex(new Vector2(x + w, y + h), col);
        dst[5] = new Vertex(new Vector2(x, y + h), col);
    }

    private void DrawMp3MenuOverlay()
    {
        if (!_mp3MenuOpen) return;

        EnsureMp3MenuResources();

        // ---- panel background ----
        _shaderColor?.Use();
        _shaderColor?.SetUniform("uResolution", new Vector2(Size.X, Size.Y));

        GL.BindVertexArray(_vaoDynamic);
        GL.BindBuffer(BufferTarget.ArrayBuffer, _vboDynamic);

        // затемнение фона (полупрозрачный слой на весь экран)
        var dim = new Vector4(0f, 0f, 0f, 0.45f);
        BuildRectVerts(0, 0, Size.X, Size.Y, dim, _mp3MenuPanelVerts);
        UploadAndDraw(_mp3MenuPanelVerts, 6, PrimitiveType.Triangles);

        // панель
        var panel = new Vector4(0.06f, 0.06f, 0.08f, 0.92f);
        BuildRectVerts(_mp3MenuX, _mp3MenuY, _mp3MenuW, _mp3MenuH, panel, _mp3MenuPanelVerts);
        UploadAndDraw(_mp3MenuPanelVerts, 6, PrimitiveType.Triangles);

        // рамка панели (тонкая)
        var brd = new Vector4(0.65f, 0.65f, 0.75f, 0.20f);
        // рисуем рамку линиями (4 сегмента)
        Vertex[] borderLines = new Vertex[8];
        float x0 = _mp3MenuX, y0 = _mp3MenuY, x1 = _mp3MenuX + _mp3MenuW, y1 = _mp3MenuY + _mp3MenuH;
        borderLines[0] = new Vertex(new Vector2(x0, y0), brd); borderLines[1] = new Vertex(new Vector2(x1, y0), brd);
        borderLines[2] = new Vertex(new Vector2(x1, y0), brd); borderLines[3] = new Vertex(new Vector2(x1, y1), brd);
        borderLines[4] = new Vertex(new Vector2(x1, y1), brd); borderLines[5] = new Vertex(new Vector2(x0, y1), brd);
        borderLines[6] = new Vertex(new Vector2(x0, y1), brd); borderLines[7] = new Vertex(new Vector2(x0, y0), brd);
        UploadAndDraw(borderLines, borderLines.Length, PrimitiveType.Lines, lineWidth: 1.25f);

        GL.BindVertexArray(0);

        // ---- text ----
        var titleTint = new Vector4(0.95f, 0.95f, 1.0f, 0.85f);
        var helpTint = new Vector4(0.90f, 0.90f, 1.0f, 0.55f);
        var itemTint = new Vector4(0.92f, 0.92f, 1.0f, 0.70f);
        var selTint = new Vector4(1.00f, 1.00f, 1.0f, 0.95f);

        // positions
        float titleX = _mp3MenuX + _mp3MenuPad;
        float titleY = _mp3MenuY + _mp3MenuPad;

        float helpX = _mp3MenuX + _mp3MenuPad;
        float helpY = _mp3MenuY + _mp3MenuH - _mp3MenuPad - _mp3MenuHelp.H;

        DrawLabel(_mp3MenuTitle, new Vector2(titleX, titleY), titleTint);
        DrawLabel(_mp3MenuHelp, new Vector2(helpX, helpY), helpTint);

        // пусто
        if (_mp3Lib.Tracks.Count == 0)
        {
            float ex = _mp3MenuX + _mp3MenuPad;
            float ey = titleY + _mp3MenuTitle.H + 18f;
            DrawLabel(_mp3MenuEmpty, new Vector2(ex, ey), itemTint);
            return;
        }

        // подсветка выбранной строки (если она видима)
        int idx = Math.Clamp(_mp3MenuIndex, 0, _mp3Lib.Tracks.Count - 1);
        Vector2 pSel = _mp3MenuItemPos[idx];
        bool selVisible = pSel.X > -1000;

        if (selVisible)
        {
            // подсветка прямоугольником
            float sx = _mp3MenuX + _mp3MenuPad - 6f;
            float sy = pSel.Y - 2f;
            float sw = _mp3MenuW - _mp3MenuPad * 2 + 12f;
            float sh = _mp3MenuRowH;

            var bgSel = new Vector4(0.25f, 0.30f, 0.45f, 0.35f);

            _shaderColor?.Use();
            _shaderColor?.SetUniform("uResolution", new Vector2(Size.X, Size.Y));
            GL.BindVertexArray(_vaoDynamic);
            GL.BindBuffer(BufferTarget.ArrayBuffer, _vboDynamic);

            BuildRectVerts(sx, sy, sw, sh, bgSel, _mp3MenuSelectVerts);
            UploadAndDraw(_mp3MenuSelectVerts, 6, PrimitiveType.Triangles);

            GL.BindVertexArray(0);
        }

        // рисуем видимые элементы списка
        for (int i = 0; i < _mp3MenuItems.Length; i++)
        {
            Vector2 pos = _mp3MenuItemPos[i];
            if (pos.X < -1000) continue;

            var tint = (i == idx) ? selTint : itemTint;
            DrawLabel(_mp3MenuItems[i], pos, tint);
        }
    }


    // Загружает выбранный трек, но НЕ запускает воспроизведение
    // Загружает выбранный трек И СРАЗУ запускает воспроизведение + визуализацию
    private void LoadSelectedMp3_Autoplay()
    {
        if (_mp3Lib.Tracks.Count == 0) return;

        _mp3MenuIndex = Math.Clamp(_mp3MenuIndex, 0, _mp3Lib.Tracks.Count - 1);
        _mp3Lib.SelectedIndex = _mp3MenuIndex;

        _mp3Lib.LastMode = AudioMode.Mp3;
        _mp3Lib.Save();

        if (_mp3 is null)
        {
            _mp3 = new Mp3AudioEngine(BarsCount, WaveSize, FftSize, MinFreq, MaxFreq);
            _mp3.TrackEnded += OnMp3TrackEnded;
        }

        // ✅ важно: autoPlay = true
        _mp3.LoadTrack(_mp3Lib.Current!, autoPlay: true);

        // ✅ визуализация включена сразу
        _vizEnabled = true;
        _particles.Clear();

        CloseMp3Menu();
        UpdateWindowTitle();
    }


    protected override void OnLoad()
    {
        base.OnLoad();
        VSync = VSyncMode.Off;

        GL.Viewport(0, 0, FramebufferSize.X, FramebufferSize.Y);

        GL.ClearColor(0f, 0f, 0f, 1f);
        GL.Disable(EnableCap.DepthTest);
        GL.Enable(EnableCap.Blend);
        GL.BlendFunc(BlendingFactor.SrcAlpha, BlendingFactor.OneMinusSrcAlpha);
        GL.Enable(EnableCap.Multisample);

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

        _loopback = new LoopbackAudioEngine(BarsCount, WaveSize, FftSize, MinFreq, MaxFreq);
        _loopback.Start();

        // если в прошлый раз был mp3 режим — восстановим (НО БЕЗ autoplay) и откроем меню
        if (_mp3Lib.LastMode == AudioMode.Mp3 && _mp3Lib.Current is not null)
        {
            _mp3 = new Mp3AudioEngine(BarsCount, WaveSize, FftSize, MinFreq, MaxFreq);
            _mp3.TrackEnded += OnMp3TrackEnded;

            // ✅ важно: НЕ играть сразу
            _mp3.LoadTrack(_mp3Lib.Current, autoPlay: false);

            _audioMode = AudioMode.Mp3;

            // ✅ чтобы первый Space запускал play
            _vizEnabled = false;
            _particles.Clear();

            // ✅ показать список треков
            LayoutMp3Menu();
            OpenMp3Menu();
        }

        UpdateWindowTitle();

    }

    private void UpdateWindowTitle()
    {
        string mode = _audioMode == AudioMode.Loopback ? "LOOPBACK" : "MP3";
        string track = (_audioMode == AudioMode.Mp3 && _mp3 is not null) ? $" | {_mp3.TrackName}" : "";
        Title = $"AudioViz [{mode}{track}] | Space: start/stop | 1-5 layers | H: fullscreen | U: hide UI | Esc: exit | M: mode";
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

        _mp3?.Dispose();
        _loopback?.Dispose();
        _mp3Lib.Save();

        if (_shaderColor is not null) _shaderColor.Dispose();
        if (_shaderBackground is not null) _shaderBackground.Dispose();

        DestroyMenuTextures();

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
        GL.Viewport(0, 0, FramebufferSize.X, FramebufferSize.Y);
        RecomputeLayout();
        LayoutMp3Menu();
        MarkMp3MenuDirty();
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
        // 1) Нижняя зона: осциллограмма
        // -----------------------------
        _waveSideMargin = w * 0.06f;

        float waveBandHeight = MathF.Max(190f, h * 0.30f);
        waveBandHeight = MathF.Min(waveBandHeight, h * 0.46f);

        _waveBandTop = h - waveBandHeight;

        // базовый padding зоны
        _wavePad = MathF.Max(6f, waveBandHeight * 0.02f);

        // границы зоны (в пикселях экрана)
        _waveYMin = _waveBandTop + _wavePad;
        _waveYMax = h - _wavePad;

        // базовая линия (чуть ниже центра)
        _waveBaseY = _waveBandTop + waveBandHeight * 0.58f;

        // half thickness ленты (ВАЖНО: будет учтён в амплитуде)
        _waveHalfTh = MathF.Max(1.2f, Size.Y * 0.0016f); // подстрой по вкусу

        // запас, чтобы ни лента, ни Catmull-овершут не упирались в край
        float headroom = _waveHalfTh + 2.0f; // +2px страховка

        // желаемая амплитуда
        float desiredAmp = waveBandHeight * 0.42f;

        // реальная доступная амплитуда с учётом headroom
        float topLimit = _waveYMin + headroom;
        float bottomLimit = _waveYMax - headroom;

        float ampTop = _waveBaseY - topLimit;
        float ampBottom = bottomLimit - _waveBaseY;

        _waveAmp = MathF.Min(desiredAmp, MathF.Max(0f, MathF.Min(ampTop, ampBottom)));

        // маленький глобальный запас (особенно помогает MP3 brickwall)
        _waveAmp *= 0.98f;

        // -----------------------------
        // 2) Верхняя зона: Stereo + Spectrum
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

        _center = new Vector2(
            spectrumX + spectrumSize * 0.5f,
            spectrumY + spectrumSize * 0.5f
        );

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

        int desired = (int)(Size.X * 1.25f);

        const float MinSrcStep = 1.6f;

        // максимальное число точек так, чтобы шаг по исходным сэмплам не стал ~1.0
        int maxByStep = (int)((WaveSize - 1) / MinSrcStep) + 1;

        // итоговый лимит
        int maxPts = Math.Min(MaxWaveDrawPoints, Math.Min(WaveSize, maxByStep));

        _waveDrawPoints = Math.Clamp(desired, MinWaveDrawPoints, maxPts);

        _waveStripVerts = new Vertex[_waveDrawPoints * 2];
        _waveLineVerts = new Vertex[_waveDrawPoints];
        _wavePts = new Vector2[_waveDrawPoints];

        _particleVerts = new Vertex[_particles.MaxParticles];
        _circleVerts = new Vertex[1 + 64 + 1];

        _stereoFrameVerts = new Vertex[16];
        _stereoScopeVerts = new Vertex[StereoPoints];

        EnsureLabels();
        UpdateLabelPositions();
        BuildCircleVertices();
    }

    private void BuildWaveTriangleStrip()
    {
        float w = Size.X;
        float margin = _waveSideMargin;
        float baseY = _waveBaseY;
        float amp = _waveAmp;

        int n = _waveDrawPoints;
        if (n < 2) return;

        // Толщина “ленты” (в пикселях)
        float thickness = MathF.Max(1.8f, Size.Y * 0.0025f); // подстрой: 0.002..0.004
        float half = thickness * 0.5f;

        // 1) строим гладкие точки центральной линии (Catmull-Rom)
        float stepSrc = (WaveSize - 1) / (float)(n - 1);

        for (int i = 0; i < n; i++)
        {
            float u = i / (float)(n - 1);
            float x = MathHelper.Lerp(margin, w - margin, u);

            float si = i * stepSrc;

            // Catmull даёт гладкость без “ступеней”
            float s = SampleWaveCatmull(si);

            // (опционально) микрофильтр против “зубцов” на высоких частотах:
            // float sPrev = SampleWaveCatmull(MathF.Max(0, si - 0.75f));
            // float sNext = SampleWaveCatmull(MathF.Min(WaveSize - 1, si + 0.75f));
            // s = 0.72f * s + 0.14f * sPrev + 0.14f * sNext;

            float y = baseY + s * amp;
            _wavePts[i] = new Vector2(x, y);
        }

        // 2) по точкам делаем triangle strip (две вершины на точку: “слева/справа” от линии)
        var col = new Vector4(0.92f, 0.92f, 1.0f, 0.90f);

        for (int i = 0; i < n; i++)
        {
            Vector2 p = _wavePts[i];

            // tangent = next - prev (стабильнее, чем next-current)
            Vector2 prev = _wavePts[Math.Max(i - 1, 0)];
            Vector2 next = _wavePts[Math.Min(i + 1, n - 1)];
            Vector2 t = next - prev;

            float len = t.Length;
            if (len < 1e-5f) t = new Vector2(1, 0);
            else t /= len;

            // normal = perp(tangent)
            Vector2 nrm = new Vector2(-t.Y, t.X);

            Vector2 off = nrm * half;

            int v = i * 2;
            _waveStripVerts[v + 0] = new Vertex(p - off, col);
            _waveStripVerts[v + 1] = new Vertex(p + off, col);
        }
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

    // добавь вверху файла (один раз):
    // using Forms = System.Windows.Forms;
    // using TkKeys = OpenTK.Windowing.GraphicsLibraryFramework.Keys;

    // ВАЖНО: вверху файла должны быть алиасы (и НЕ должно быть `using System.Windows.Forms;`)
    // using Forms = System.Windows.Forms;
    // using TkKeys = OpenTK.Windowing.GraphicsLibraryFramework.Keys;

    protected override void OnKeyDown(KeyboardKeyEventArgs e)
    {
        base.OnKeyDown(e);

        // ---------- MP3 Library toggle (Tab) ----------
        if (e.Key == TkKeys.Tab)
        {
            // Открываем/закрываем библиотеку, НЕ меняя режимы
            // (работает именно в MP3 режиме, чтобы можно было менять песни без M)
            if (_audioMode == AudioMode.Mp3)
            {
                if (_mp3MenuOpen) CloseMp3Menu();
                else
                {
                    LayoutMp3Menu();
                    OpenMp3Menu();
                }
            }
            return;
        }

        // ---------- ESC ----------
        if (e.Key == TkKeys.Escape)
        {
            if (_audioMode == AudioMode.Mp3 && _mp3MenuOpen)
            {
                CloseMp3Menu();
                return;
            }

            Close();
            return;
        }

        // ---------- Fullscreen ----------
        if (e.Key == TkKeys.H)
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

            LayoutMp3Menu();
            MarkMp3MenuDirty();
            return;
        }

        // ---------- Hide UI ----------
        if (e.Key == TkKeys.U)
        {
            _hideUi = !_hideUi;
            return;
        }

        // ---------- Layers ----------
        if (e.Key == TkKeys.D1 || e.Key == TkKeys.KeyPad1) { _layerSpectrum = !_layerSpectrum; return; }
        if (e.Key == TkKeys.D2 || e.Key == TkKeys.KeyPad2) { _layerWave = !_layerWave; return; }
        if (e.Key == TkKeys.D3 || e.Key == TkKeys.KeyPad3) { _layerParticles = !_layerParticles; return; }
        if (e.Key == TkKeys.D4 || e.Key == TkKeys.KeyPad4) { _layerBackground = !_layerBackground; return; }
        if (e.Key == TkKeys.D5 || e.Key == TkKeys.KeyPad5) { _layerStereo = !_layerStereo; return; }

        // ---------- MODE toggle (M) ----------
        if (e.Key == TkKeys.M)
        {
            if (_audioMode == AudioMode.Loopback)
            {
                // В MP3 режим — НЕ играть, открыть меню
                _audioMode = AudioMode.Mp3;
                _mp3Lib.LastMode = AudioMode.Mp3;
                _mp3Lib.Save();

                _mp3?.Pause();          // если что-то было загружено ранее — не играть
                _vizEnabled = false;
                _particles.Clear();
                LayoutMp3Menu();
                OpenMp3Menu();          // внутри ставит dirty + title
            }
            else
            {
                // Назад в Loopback
                _mp3?.Pause();
                CloseMp3Menu();

                _audioMode = AudioMode.Loopback;
                _mp3Lib.LastMode = AudioMode.Loopback;
                _mp3Lib.Save();

                UpdateWindowTitle();
            }

            return;
        }

        // ---------- IMPORT MP3 (O) ----------
        if (e.Key == TkKeys.O)
        {
            try
            {
                using var ofd = new Forms.OpenFileDialog
                {
                    Filter = "MP3 files (*.mp3)|*.mp3",
                    Title = "Import MP3 to AudioViz library",
                    Multiselect = false
                };

                if (ofd.ShowDialog() == Forms.DialogResult.OK)
                {
                    _mp3Lib.ImportToLibrary(ofd.FileName);
                    _mp3Lib.LastMode = AudioMode.Mp3;
                    _mp3Lib.Save();

                    // переключаемся в MP3 режим и открываем меню (без autoplay)
                    _audioMode = AudioMode.Mp3;
                    _mp3?.Pause();

                    LayoutMp3Menu();
                    OpenMp3Menu();

                    // выделим импортированный трек (ImportToLibrary обычно делает SelectedIndex на новый)
                    _mp3MenuIndex = _mp3Lib.SelectedIndex;
                    MarkMp3MenuDirty();
                }
            }
            catch (Exception ex)
            {
                Debug.WriteLine(ex);
            }

            return;
        }

        // ============================================================
        //                     MP3 MENU (если открыто)
        // ============================================================
        if (_audioMode == AudioMode.Mp3 && _mp3MenuOpen)
        {
            int n = _mp3Lib.Tracks.Count;

            if (e.Key == TkKeys.Up)
            {
                if (n > 0)
                {
                    _mp3MenuIndex = (_mp3MenuIndex - 1 + n) % n;
                    MarkMp3MenuDirty(); // чтобы пересчитались позиции/окно списка
                }
                return;
            }

            if (e.Key == TkKeys.Down)
            {
                if (n > 0)
                {
                    _mp3MenuIndex = (_mp3MenuIndex + 1) % n;
                    MarkMp3MenuDirty();
                }
                return;
            }

            if (e.Key == TkKeys.Home)
            {
                if (n > 0)
                {
                    _mp3MenuIndex = 0;
                    MarkMp3MenuDirty();
                }
                return;
            }

            if (e.Key == TkKeys.End)
            {
                if (n > 0)
                {
                    _mp3MenuIndex = n - 1;
                    MarkMp3MenuDirty();
                }
                return;
            }

            // Enter = загрузить выбранный (БЕЗ play) и закрыть меню
            // Enter = загрузить выбранный И СРАЗУ играть + визуализация
            if (e.Key == TkKeys.Enter || e.Key == TkKeys.KeyPadEnter)
            {
                if (n > 0)
                {
                    _mp3MenuIndex = Math.Clamp(_mp3MenuIndex, 0, n - 1);
                    LoadSelectedMp3_Autoplay();   // ✅ autoplay
                    MarkMp3MenuDirty();
                }
                return;
            }

            // Delete = удалить выбранный из библиотеки (и файл из папки библиотеки)
            if (e.Key == TkKeys.Delete)
            {
                if (n > 0)
                {
                    _mp3MenuIndex = Math.Clamp(_mp3MenuIndex, 0, n - 1);

                    string removedPath = _mp3Lib.Tracks[_mp3MenuIndex];

                    // если удаляем текущий загруженный — остановим/выкинем
                    if (_mp3 is not null && string.Equals(_mp3.TrackPath, removedPath, StringComparison.OrdinalIgnoreCase))
                    {
                        _mp3.Stop();
                        _mp3 = null;
                    }

                    _mp3Lib.SelectedIndex = _mp3MenuIndex;
                    _mp3Lib.RemoveCurrent();
                    _mp3Lib.Save();

                    int nn = _mp3Lib.Tracks.Count;
                    _mp3MenuIndex = (nn == 0) ? 0 : Math.Clamp(_mp3MenuIndex, 0, nn - 1);

                    MarkMp3MenuDirty();
                    UpdateWindowTitle();
                }
                return;
            }

            // Space в меню игнорируем (чтобы ничего случайно не включить)
            if (e.Key == TkKeys.Space)
                return;

            // Любые другие клавиши в меню не обрабатываем
            return;
        }

        // ============================================================
        //                 Обычное управление (вне меню)
        // ============================================================

        // Space: как раньше + в MP3 режиме play/pause (если трек загружен)
        if (e.Key == TkKeys.Space)
        {
            _vizEnabled = !_vizEnabled;

            if (_audioMode == AudioMode.Mp3 && _mp3 is not null)
            {
                if (_vizEnabled) _mp3.Play();
                else _mp3.Pause();
            }

            return;
        }

        // В MP3 режиме (вне меню): открыть меню на J/K (удобно)
        if (_audioMode == AudioMode.Mp3 && (e.Key == TkKeys.J || e.Key == TkKeys.K))
        {
            LayoutMp3Menu();
            OpenMp3Menu();
            return;
        }

        // Enter вне меню: restart (опционально)
        if (e.Key == TkKeys.Enter || e.Key == TkKeys.KeyPadEnter)
        {
            if (_audioMode == AudioMode.Mp3 && _mp3 is not null)
                _mp3.Restart();
            return;
        }
    }

    protected override void OnUpdateFrame(FrameEventArgs args)
    {
        base.OnUpdateFrame(args);

        _timeSec += args.Time;
        // --- MP3 autoplay next track when current ended ---
        if (_audioMode == AudioMode.Mp3 && _mp3 is not null &&
            Interlocked.Exchange(ref _mp3AutoNextFlag, 0) == 1)
        {
            if (_mp3Lib.Tracks.Count > 0)
            {
                _mp3Lib.Next();               // уже зациклено: после последнего -> первый
                _mp3Lib.LastMode = AudioMode.Mp3;
                _mp3Lib.Save();

                _vizEnabled = true;           // чтобы точно играло
                _particles.Clear();

                _mp3.LoadTrack(_mp3Lib.Current!, autoPlay: true);

                // если меню открыто — подсветим новый трек
                _mp3MenuIndex = _mp3Lib.SelectedIndex;
                MarkMp3MenuDirty();
                UpdateWindowTitle();
            }
        }

        // если MP3 режим, но трек еще не выбран/не загружен — рисуем тишину
        if (_audioMode == AudioMode.Mp3 && _mp3 is null)
        {
            Array.Clear(_barTargets, 0, _barTargets.Length);
            Array.Clear(_wave, 0, _wave.Length);
            Array.Clear(_stL, 0, _stL.Length);
            Array.Clear(_stR, 0, _stR.Length);
            _particles.Clear();
            return;
        }
        Audio.Update((float)args.Time);

        if (_audioMode == AudioMode.Mp3 && _mp3 is not null && _mp3.ConsumeTrackEnded())
        {
            PlayNextMp3_Autoplay();
        }

        // Pull audio snapshots
        Audio.CopyBars(_barTargets);
        Audio.CopyWaveform(_wave);
        // убрать DC offset (особенно заметно на MP3)
        float mean = 0f;
        for (int i = 0; i < _wave.Length; i++) mean += _wave[i];
        mean /= _wave.Length;
        for (int i = 0; i < _wave.Length; i++) _wave[i] -= mean;

        if (_audioMode == AudioMode.Mp3)
            SmoothWaveForDisplay(_wave);

        UpdateWaveDisplayGain((float)args.Time);

        // Stereo snapshot (L/R)
        if (_layerStereo)
        {
            Audio.CopyStereo(_tmpL, _tmpR);
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
        while (Audio.TryDequeueBeat(out var beat))
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

            float bpm = Audio.BpmSmoothed;
            float speed = MathHelper.Clamp(bpm / 120f, 0.15f, 3.0f);

            float beatEnv = MathF.Exp(-(float)(_timeSec - _lastBeatFlash) / 0.45f);
            float beatGlow = beatEnv * (0.25f + 0.75f * _lastBeatStrength);

            _shaderBackground.SetUniform("uTime", (float)_timeSec);
            _shaderBackground.SetUniform("uSpeed", speed);
            _shaderBackground.SetUniform("uBeatGlow", beatGlow);
            _shaderBackground.SetUniform("uBeatTint", _lastBeatColor);

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

        // ---- Oscilloscope as smooth ribbon (TriangleStrip + LineStrip) ----
        if (_layerWave && _waveDrawPoints > 3)
        {
            // Требуются поля:
            // Vertex[] _waveStripVerts;  // size = _waveDrawPoints * 2
            // Vertex[] _waveLineVerts;   // size = _waveDrawPoints
            // Vector2[] _wavePts;        // size = _waveDrawPoints

            void BuildWaveRibbon()
            {
                int n = _waveDrawPoints;
                if (n < 2) return;

                float width = Size.X;
                float margin = _waveSideMargin;
                float baseY = _waveBaseY;
                float amp = _waveAmp;

                float halfTh = MathF.Max(1.0f, _waveHalfTh);

                var colFill = new Vector4(0.92f, 0.92f, 1.00f, 0.22f);
                var colLine = new Vector4(0.92f, 0.92f, 1.00f, 0.88f);

                // границы зоны + запас под толщину
                float yMin = _waveYMin + halfTh + 1.0f;
                float yMax = _waveYMax - halfTh - 1.0f;

                float spanX = MathF.Max(1.0f, width - 2f * margin);
                float srcMax = WaveSize - 1;

                // шаг для производной (в индексах сэмплов)
                const float dSrc = 1.0f;

                for (int i = 0; i < n; i++)
                {
                    float u = i / (float)(n - 1);
                    float x = margin + u * spanX;

                    float src = u * srcMax;

                    // центральная линия
                    float s = SampleWaveCatmull(src) * _waveDisplayGain;

                    // лучше не давать Catmull "перелетать" вообще
                    s = MathHelper.Clamp(s, -1f, 1f);

                    // (опционально) ещё красивее: мягкая компрессия пиков вместо жёсткого clamp
                    // s = MathF.Tanh(s * 1.35f);

                    float y = baseY + s * amp;
                    y = Math.Clamp(y, yMin, yMax);

                    Vector2 p = new Vector2(x, y);
                    _wavePts[i] = p;
                    _waveLineVerts[i] = new Vertex(p, colLine);

                    // ---- стабильная нормаль через оценку наклона dy/dx от исходной волны ----
                    float s0 = SampleWaveCatmull(MathF.Max(0f, src - dSrc));
                    float s1 = SampleWaveCatmull(MathF.Min(srcMax, src + dSrc));
                    float ds_dsrc = (s1 - s0) / (2f * dSrc);

                    // dy/du = amp * ds_dsrc * (WaveSize-1); dx/du = spanX
                    float dy_dx = (amp * ds_dsrc * srcMax) / spanX;
                    dy_dx = Math.Clamp(dy_dx, -8f, 8f); // страховка от экстремальных наклонов

                    Vector2 nrm = new Vector2(-dy_dx, 1f);
                    float nl = nrm.Length;
                    if (nl < 1e-5f) nrm = new Vector2(0f, 1f);
                    else nrm /= nl;

                    Vector2 off = nrm * halfTh;

                    int v = i * 2;
                    _waveStripVerts[v + 0] = new Vertex(p - off, colFill);
                    _waveStripVerts[v + 1] = new Vertex(p + off, colFill);
                }
            }

            BuildWaveRibbon();

            // лента (мягкая заливка)
            UploadAndDraw(_waveStripVerts, _waveDrawPoints * 2, PrimitiveType.TriangleStrip);

            // контур поверх (чтобы “волна” читалась чётко)
            GL.BlendFunc(BlendingFactor.SrcAlpha, BlendingFactor.One);
            UploadAndDraw(_waveLineVerts, _waveDrawPoints, PrimitiveType.LineStrip, lineWidth: 1.6f);
            GL.BlendFunc(BlendingFactor.SrcAlpha, BlendingFactor.OneMinusSrcAlpha);
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
        if (!_hideUi && _shaderText is not null)
        {
            var tint = new Vector4(0.90f, 0.90f, 1.00f, 0.55f);

            if (_layerStereo) DrawLabel(_lblStereo, _lblStereoPos, tint);
            if (_layerSpectrum) DrawLabel(_lblSpectrum, _lblSpectrumPos, tint);
            if (_layerWave) DrawLabel(_lblWave, _lblWavePos, tint);
        }

        if (_audioMode == AudioMode.Mp3 && _mp3MenuOpen)
        {
            DrawMp3MenuOverlay();
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

    private static float CatmullRom(float p0, float p1, float p2, float p3, float t)
    {
        // t in [0..1]
        float t2 = t * t;
        float t3 = t2 * t;

        return 0.5f * (
            2f * p1 +
            (-p0 + p2) * t +
            (2f * p0 - 5f * p1 + 4f * p2 - p3) * t2 +
            (-p0 + 3f * p1 - 3f * p2 + p3) * t3
        );
    }

    private float SampleWaveCatmull(float srcIndex)
    {
        // srcIndex in [0..WaveSize-1]
        int i1 = (int)srcIndex;
        float t = srcIndex - i1;

        int i0 = Math.Max(i1 - 1, 0);
        int i2 = Math.Min(i1 + 1, WaveSize - 1);
        int i3 = Math.Min(i1 + 2, WaveSize - 1);

        return CatmullRom(_wave[i0], _wave[i1], _wave[i2], _wave[i3], t);
    }

    private int BuildWaveEnvelopeVertices()
    {
        float w = Size.X;
        float margin = _waveSideMargin;
        float baseY = _waveBaseY;
        float amp = _waveAmp;

        var col = new Vector4(0.92f, 0.92f, 1.0f, 0.85f);

        int cols = _waveDrawPoints;
        if (cols <= 1) return 0;

        float step = _wave.Length / (float)cols; // тут важно: этот метод использовать только когда step >= 1

        int v = 0;
        for (int i = 0; i < cols; i++)
        {
            float u = i / (float)(cols - 1);
            float x = MathHelper.Lerp(margin, w - margin, u);

            int s0 = (int)(i * step);
            int s1 = (int)((i + 1) * step);
            if (s1 <= s0) s1 = s0 + 1;
            if (s1 > _wave.Length) s1 = _wave.Length;

            float mn = float.PositiveInfinity;
            float mx = float.NegativeInfinity;

            for (int s = s0; s < s1; s++)
            {
                float a = _wave[s];
                if (a < mn) mn = a;
                if (a > mx) mx = a;
            }

            float y0 = baseY + mn * amp;
            float y1 = baseY + mx * amp;

            _waveVerts[v++] = new Vertex(new Vector2(x, y0), col);
            _waveVerts[v++] = new Vertex(new Vector2(x, y1), col);
        }

        return v; // cols*2
    }

    private int BuildWaveLineStripVertices()
    {
        float w = Size.X;
        float margin = _waveSideMargin;
        float baseY = _waveBaseY;
        float amp = _waveAmp;

        var col = new Vector4(0.92f, 0.92f, 1.0f, 0.85f);

        int cols = _waveDrawPoints;
        if (cols <= 1) return 0;

        for (int i = 0; i < cols; i++)
        {
            float u = i / (float)(cols - 1);
            float x = MathHelper.Lerp(margin, w - margin, u);

            // индекс по исходной волне 0..WaveSize-1
            float src = u * (WaveSize - 1);
            float a = SampleWaveCatmull(src);   // уже есть у тебя
            float y = baseY + a * amp;

            _waveLineVerts[i] = new Vertex(new Vector2(x, y), col);
        }

        return cols;
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

        if (prim == PrimitiveType.LineStrip || prim == PrimitiveType.Lines)
            GL.LineWidth(lineWidth);
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

#region Audio Modes + MP3 Library + Engines

internal enum AudioMode
{
    Loopback = 0,
    Mp3 = 1
}

internal interface IAudioEngine : IDisposable
{
    AudioMode Mode { get; }
    float BpmSmoothed { get; }

    void Start();
    void Stop();

    // Для MP3 режима: синхронизация вывода под позицию воспроизведения
    void Update(float dt);

    void CopyBars(float[] dest);
    void CopyWaveform(float[] dest);
    void CopyStereo(float[] left, float[] right);

    bool TryDequeueBeat(out BeatEvent beat);
}

// ====== Settings + library (файлы реально сохраняются локально) ======

internal sealed class Mp3Library
{
    private sealed class Settings
    {
        public List<string> Tracks { get; set; } = new();
        public int SelectedIndex { get; set; } = 0;
        public AudioMode LastMode { get; set; } = AudioMode.Loopback;
    }

    private Settings _s = new();

    public IReadOnlyList<string> Tracks => _s.Tracks;
    public int SelectedIndex
    {
        get => (_s.Tracks.Count == 0) ? 0 : Math.Clamp(_s.SelectedIndex, 0, _s.Tracks.Count - 1);
        set => _s.SelectedIndex = (_s.Tracks.Count == 0) ? 0 : Math.Clamp(value, 0, _s.Tracks.Count - 1);
    }

    public AudioMode LastMode
    {
        get => _s.LastMode;
        set => _s.LastMode = value;
    }

    public string? Current => (_s.Tracks.Count == 0) ? null : _s.Tracks[SelectedIndex];

    private static string AppDir =>
        Path.Combine(Environment.GetFolderPath(Environment.SpecialFolder.ApplicationData), "AudioViz");

    private static string TracksDir => Path.Combine(AppDir, "Tracks");
    private static string SettingsPath => Path.Combine(AppDir, "library.json");

    public static Mp3Library Load()
    {
        var lib = new Mp3Library();
        try
        {
            Directory.CreateDirectory(AppDir);
            Directory.CreateDirectory(TracksDir);

            if (File.Exists(SettingsPath))
            {
                var json = File.ReadAllText(SettingsPath);
                var s = JsonSerializer.Deserialize<Settings>(json);
                if (s is not null) lib._s = s;
            }

            // выкинуть отсутствующие
            lib._s.Tracks = lib._s.Tracks.Where(File.Exists).ToList();
            lib.SelectedIndex = lib._s.SelectedIndex;
        }
        catch
        {
            // молча: если settings битые, просто стартуем с пустого
            lib._s = new Settings();
        }
        return lib;
    }

    public void Save()
    {
        try
        {
            Directory.CreateDirectory(AppDir);
            Directory.CreateDirectory(TracksDir);

            var json = JsonSerializer.Serialize(_s, new JsonSerializerOptions { WriteIndented = true });
            File.WriteAllText(SettingsPath, json);
        }
        catch { }
    }

    public string ImportToLibrary(string sourceFile)
    {
        Directory.CreateDirectory(TracksDir);

        string baseName = Path.GetFileName(sourceFile);
        string nameNoExt = Path.GetFileNameWithoutExtension(baseName);
        string ext = Path.GetExtension(baseName);

        if (!ext.Equals(".mp3", StringComparison.OrdinalIgnoreCase))
            throw new InvalidOperationException("Only .mp3 supported in this import method.");

        string dest = Path.Combine(TracksDir, baseName);
        int n = 1;
        while (File.Exists(dest))
        {
            dest = Path.Combine(TracksDir, $"{nameNoExt} ({n}){ext}");
            n++;
        }

        File.Copy(sourceFile, dest, overwrite: false);

        _s.Tracks.Add(dest);
        SelectedIndex = _s.Tracks.Count - 1;
        Save();
        return dest;
    }

    public void Next()
    {
        if (_s.Tracks.Count == 0) return;
        SelectedIndex = (SelectedIndex + 1) % _s.Tracks.Count;
        Save();
    }

    public void Prev()
    {
        if (_s.Tracks.Count == 0) return;
        SelectedIndex = (SelectedIndex - 1 + _s.Tracks.Count) % _s.Tracks.Count;
        Save();
    }

    public void RemoveCurrent()
    {
        if (_s.Tracks.Count == 0) return;
        string p = _s.Tracks[SelectedIndex];
        _s.Tracks.RemoveAt(SelectedIndex);
        SelectedIndex = Math.Min(SelectedIndex, _s.Tracks.Count - 1);
        Save();

        // Если файл лежит в нашей папке Tracks — удалим физически (чтобы не копить мусор)
        try
        {
            if (p.StartsWith(TracksDir, StringComparison.OrdinalIgnoreCase) && File.Exists(p))
                File.Delete(p);
        }
        catch { }
    }
}

// ====== Loopback wrapper: ничего не меняем в твоём AudioAnalyzer ======

internal sealed class LoopbackAudioEngine : IAudioEngine
{
    private readonly AudioAnalyzer _a;
    public AudioMode Mode => AudioMode.Loopback;
    public float BpmSmoothed => _a.BpmSmoothed;

    public LoopbackAudioEngine(int barsCount, int waveSize, int fftSize, float minFreq, float maxFreq)
    {
        _a = new AudioAnalyzer(barsCount, waveSize, fftSize, minFreq, maxFreq);
    }

    public void Start() => _a.Start();
    public void Stop() => Dispose();
    public void Update(float dt) { /* no-op */ }

    public void CopyBars(float[] dest) => _a.CopyBars(dest);
    public void CopyWaveform(float[] dest) => _a.CopyWaveform(dest);
    public void CopyStereo(float[] left, float[] right) => _a.CopyStereo(left, right);

    public bool TryDequeueBeat(out BeatEvent beat) => _a.TryDequeueBeat(out beat);

    public void Dispose() => _a.Dispose();
}

// ====== MP3 Engine: предобработка + синхронизация по playback position ======

internal sealed class Mp3AudioEngine : IAudioEngine
{
    public AudioMode Mode => AudioMode.Mp3;

    private readonly int _barsCount;
    private readonly int _waveSize;
    private readonly int _fftSize;
    private readonly float _minFreq;
    private readonly float _maxFreq;

    private long _playPosWave;          // кэш позиции для wave/stereo (сглаженный)
    private double _playPosWaveF;       // float-версия для EMA
    private bool _playPosWaveInit;

    private int _endedFlag = 0;

    private int _sr;
    private int _latencyFrames;
    private const int WasapiLatencyMs = 90; // у тебя ровно это значение в WasapiOut(...)

    public event Action? TrackEnded;

    private int _endSignaled = 0;     // чтобы сработало один раз
    private int _suppressEnd = 1;     // 1 = не триггерим end (во время Stop/переключений)

    // bars/bpm for visualizer
    private readonly object _barsLock = new();
    private readonly float[] _barsFront;
    private float _bpmFront = 120f;

    // beats out
    private readonly ConcurrentQueue<BeatEvent> _beatOut = new();

    // ring buffers for wave + stereo
    private readonly object _ringLock = new();
    private float[] _monoRing = Array.Empty<float>();
    private float[] _lRing = Array.Empty<float>();
    private float[] _rRing = Array.Empty<float>();
    private int _ringCapFrames;
    private long _framesClock; // how many FRAMES were actually read to output chain

    // playback
    private WasapiOut? _out;
    private AudioFileReader? _reader;
    private ISampleProvider? _source;
    private TapSampleProvider? _tap;

    private StreamSpectrumAnalyzer? _an;

    public string TrackPath { get; private set; } = "";
    public string TrackName => string.IsNullOrWhiteSpace(TrackPath) ? "No track" : Path.GetFileNameWithoutExtension(TrackPath);

    public float BpmSmoothed
    {
        get { lock (_barsLock) return _bpmFront; }
    }

    internal void SignalTrackEnded()
    {
        Interlocked.Exchange(ref _endedFlag, 1);
    }

    public bool ConsumeTrackEnded()
    {
        return Interlocked.Exchange(ref _endedFlag, 0) == 1;
    }

    public Mp3AudioEngine(int barsCount, int waveSize, int fftSize, float minFreq, float maxFreq)
    {
        _barsCount = barsCount;
        _waveSize = waveSize;
        _fftSize = fftSize;
        _minFreq = minFreq;
        _maxFreq = maxFreq;

        _barsFront = new float[_barsCount];
    }

    public void LoadTrack(string path, bool autoPlay = true)
    {
        Stop();

        TrackPath = path;
        _playPosWave = 0;
        _playPosWaveF = 0;
        _playPosWaveInit = false;

        // reader -> float PCM
        _reader = new AudioFileReader(path);
        _source = _reader;

        // make stereo if mono
        if (_source.WaveFormat.Channels == 1)
            _source = new MonoToStereoSampleProvider(_source);

        // (опционально, но помогает стабильности) подгоним sample rate под mix-format устройства
        try
        {
            using var dev = new MMDeviceEnumerator().GetDefaultAudioEndpoint(DataFlow.Render, Role.Multimedia);
            int mixSr = dev.AudioClient.MixFormat.SampleRate;

            if (_source.WaveFormat.SampleRate != mixSr)
                _source = new WdlResamplingSampleProvider(_source, mixSr);
        }
        catch
        {
            // если CoreAudio недоступен — просто оставим исходный SR
        }

        int sr = _source.WaveFormat.SampleRate;
        int ch = _source.WaveFormat.Channels;

        _sr = sr;
        _latencyFrames = (int)Math.Round(_sr * (WasapiLatencyMs / 1000.0));
        Interlocked.Exchange(ref _framesClock, 0);

        // ring buffers (20 seconds window)
        _ringCapFrames = Math.Max(sr * 20, _waveSize + 1024);
        _monoRing = new float[_ringCapFrames];
        _lRing = new float[_ringCapFrames];
        _rRing = new float[_ringCapFrames];
        _framesClock = 0;

        // analyzer
        _an = new StreamSpectrumAnalyzer(_barsCount, _fftSize, _minFreq, _maxFreq, sr);

        _an.OnBarsFrame += (_, bars, bpm) =>
        {
            lock (_barsLock)
            {
                Array.Copy(bars, _barsFront, _barsCount);
                _bpmFront = bpm;
            }
        };

        _an.OnBeatFrame += (_, beat) =>
        {
            _beatOut.Enqueue(beat);
        };

        // небольшой запас по уровню, чтобы MP3/resampler не клиппили пики
        _source = new VolumeSampleProvider(_source) { Volume = 0.90f };

        // tap: теперь видит уже “безопасные” сэмплы (и звук, и визуал совпадают)
        _tap = new TapSampleProvider(_source, ch, this);

        // output
        _out = new WasapiOut(AudioClientShareMode.Shared, useEventSync: true, latency: 90);
        _out.PlaybackStopped += Out_PlaybackStopped;

        // ⚠️ ВАЖНО: не 16-bit, а float
        var waveProvider = new SampleToWaveProvider(_tap);
        _out.Init(waveProvider);

        // новый трек -> можно снова ловить окончание
        Interlocked.Exchange(ref _endSignaled, 0);
        Interlocked.Exchange(ref _suppressEnd, 0);

        if (autoPlay) _out.Play();
        else _out.Pause();
    }
    private long GetPlaybackTargetFrames()
    {
        long produced = Interlocked.Read(ref _framesClock); // сколько фреймов отдали в output-цепочку
        long target = produced - _latencyFrames;            // приблизительно сколько уже "прозвучало"
        return target < 0 ? 0 : target;
    }

    public void Start() { /* no-op */ }

    private void Out_PlaybackStopped(object? sender, StoppedEventArgs e)
    {
        // не реагируем на Stop()/переключения/удаления/ручные остановки
        if (Interlocked.CompareExchange(ref _suppressEnd, 0, 0) == 1)
            return;

        if (_reader is null)
            return;

        // если это не "дошли до конца файла", а остановили по другой причине — игнор
        // (позиция иногда не ровно в конец, поэтому небольшой допуск)
        long slack = _reader.WaveFormat.BlockAlign * 8;
        bool ended = _reader.Position >= (_reader.Length - slack);
        if (!ended)
            return;

        // гарантируем одно срабатывание на трек
        if (Interlocked.Exchange(ref _endSignaled, 1) == 0)
            TrackEnded?.Invoke();
    }

    public void Stop()
    {
        Interlocked.Exchange(ref _suppressEnd, 1);
        Interlocked.Exchange(ref _endedFlag, 0);

        try { _out?.Stop(); } catch { }

        if (_out is not null)
        {
            try { _out.PlaybackStopped -= Out_PlaybackStopped; } catch { }
            try { _out.Dispose(); } catch { }
            _out = null;
        }

        if (_reader is not null)
        {
            try { _reader.Dispose(); } catch { }
            _reader = null;
        }

        _source = null;
        _tap = null;
        _an = null;

        while (_beatOut.TryDequeue(out _)) { }

        lock (_barsLock)
        {
            Array.Clear(_barsFront, 0, _barsFront.Length);
            _bpmFront = 120f;
        }

        lock (_ringLock)
        {
            if (_monoRing.Length > 0) Array.Clear(_monoRing, 0, _monoRing.Length);
            if (_lRing.Length > 0) Array.Clear(_lRing, 0, _lRing.Length);
            if (_rRing.Length > 0) Array.Clear(_rRing, 0, _rRing.Length);
            _framesClock = 0;
        }

        TrackPath = "";
        _playPosWave = 0;
        _playPosWaveF = 0;
        _playPosWaveInit = false;
        Interlocked.Exchange(ref _endedFlag, 0);
    }

    public void Dispose() => Stop();

    public void Pause() => _out?.Pause();
    public void Play() => _out?.Play();

    public void Restart()
    {
        if (string.IsNullOrWhiteSpace(TrackPath)) return;
        LoadTrack(TrackPath, autoPlay: true);
    }

    public void Update(float dt)
    {
        if (_sr <= 0) return;

        long target = GetPlaybackTargetFrames();

        if (!_playPosWaveInit)
        {
            _playPosWaveInit = true;
            _playPosWaveF = target;
        }
        else
        {
            // 1) предсказываем плавное движение по времени (чтобы не "скакало" пачками)
            if (_out?.PlaybackState == PlaybackState.Playing)
                _playPosWaveF += dt * _sr;

            // 2) мягко подтягиваемся к реальному target (коррекция дрейфа)
            double pull = 1.0 - Math.Exp(-dt / 0.080); // 80ms
            _playPosWaveF += (target - _playPosWaveF) * pull;

            // 3) не обгоняем реально доступные данные
            if (_playPosWaveF > target) _playPosWaveF = target;
            if (_playPosWaveF < 0) _playPosWaveF = 0;
        }

        _playPosWave = (long)Math.Round(_playPosWaveF);
    }

    public bool TryDequeueBeat(out BeatEvent beat) => _beatOut.TryDequeue(out beat);

    public void CopyBars(float[] dest)
    {
        if (dest.Length != _barsCount) throw new ArgumentException("bars array size mismatch.");
        lock (_barsLock) Array.Copy(_barsFront, dest, _barsCount);
    }

    public void CopyWaveform(float[] dest)
    {
        if (dest.Length != _waveSize) throw new ArgumentException("wave array size mismatch.");

        long playPos = _playPosWave;
        long framesWritten = Interlocked.Read(ref _framesClock);

        lock (_ringLock)
        {
            long oldest = Math.Max(0, framesWritten - _ringCapFrames);
            long newest = framesWritten - 1;

            long start = playPos - _waveSize;

            int prefix = 0;
            if (start < oldest)
            {
                prefix = (int)Math.Min((long)_waveSize, oldest - start);
                float fill = (newest >= 0) ? _monoRing[(int)(oldest % _ringCapFrames)] : 0f;
                for (int i = 0; i < prefix; i++) dest[i] = fill;
                start = oldest;
            }

            for (int i = prefix; i < _waveSize; i++)
            {
                long idxAbs = start + (i - prefix);

                if (idxAbs < oldest || idxAbs > newest)
                {
                    dest[i] = (newest >= 0) ? _monoRing[(int)(newest % _ringCapFrames)] : 0f;
                    continue;
                }

                int k = (int)(idxAbs % _ringCapFrames);
                dest[i] = _monoRing[k];
            }
        }
    }
    public void CopyStereo(float[] left, float[] right)
    {
        if (left.Length != _waveSize || right.Length != _waveSize)
            throw new ArgumentException("stereo arrays size mismatch.");

        long playPos = _playPosWave;
        long framesWritten = Interlocked.Read(ref _framesClock);

        lock (_ringLock)
        {
            long oldest = Math.Max(0, framesWritten - _ringCapFrames);
            long newest = framesWritten - 1;

            long start = playPos - _waveSize;

            int prefix = 0;
            if (start < oldest)
            {
                prefix = (int)Math.Min((long)_waveSize, oldest - start);
                float fillL = (newest >= 0) ? _lRing[(int)(oldest % _ringCapFrames)] : 0f;
                float fillR = (newest >= 0) ? _rRing[(int)(oldest % _ringCapFrames)] : 0f;

                for (int i = 0; i < prefix; i++) { left[i] = fillL; right[i] = fillR; }
                start = oldest;
            }

            for (int i = prefix; i < _waveSize; i++)
            {
                long idxAbs = start + (i - prefix);

                if (idxAbs < oldest || idxAbs > newest)
                {
                    float fillL = (newest >= 0) ? _lRing[(int)(newest % _ringCapFrames)] : 0f;
                    float fillR = (newest >= 0) ? _rRing[(int)(newest % _ringCapFrames)] : 0f;
                    left[i] = fillL;
                    right[i] = fillR;
                    continue;
                }

                int k = (int)(idxAbs % _ringCapFrames);
                left[i] = _lRing[k];
                right[i] = _rRing[k];
            }
        }
    }

    // called from TapSampleProvider
    private void OnOutputSamples(float[] buffer, int offset, int frames, int channels)
    {
        if (_an is null) return;

        long startFrame = System.Threading.Interlocked.Read(ref _framesClock);

        lock (_ringLock)
        {
            for (int f = 0; f < frames; f++)
            {
                float L = buffer[offset + f * channels + 0];
                float R = (channels > 1) ? buffer[offset + f * channels + 1] : L;
                float mono = (L + R) * 0.5f;

                int k = (int)((startFrame + f) % _ringCapFrames);
                _monoRing[k] = mono;
                _lRing[k] = L;
                _rRing[k] = R;

                _an.PushMonoSample(mono);
            }
        }

        System.Threading.Interlocked.Add(ref _framesClock, frames);
    }

    private sealed class TapSampleProvider : ISampleProvider
    {
        private readonly ISampleProvider _src;
        private readonly int _channels;
        private readonly Mp3AudioEngine _owner;

        public TapSampleProvider(ISampleProvider src, int channels, Mp3AudioEngine owner)
        {
            _src = src;
            _channels = channels;
            _owner = owner;
        }

        public WaveFormat WaveFormat => _src.WaveFormat;

        public int Read(float[] buffer, int offset, int count)
        {
            int n = _src.Read(buffer, offset, count);
            if (n <= 0)
            {
                _owner.SignalTrackEnded();
                return 0;
            }

            // выравниваем до целых фреймов
            n -= n % _channels;
            if (n <= 0) return 0;

            int frames = n / _channels;
            _owner.OnOutputSamples(buffer, offset, frames, _channels);

            return n;
        }
    }

    // ====== StreamSpectrumAnalyzer (оставь как у тебя, без изменений) ======
    // Ниже — твой же класс, просто перенесён как есть (можешь вставить из своего файла).
    private sealed class StreamSpectrumAnalyzer
    {
        public delegate void BarsFrameHandler(long samplePos, float[] bars, float bpm);
        public delegate void BeatFrameHandler(long samplePos, BeatEvent beat);

        public event BarsFrameHandler? OnBarsFrame;
        public event BeatFrameHandler? OnBeatFrame;

        private readonly int _barsCount;
        private readonly int _fftSize;
        private readonly float _minFreq;
        private readonly float _maxFreq;
        private readonly int _sampleRate;

        private readonly int _hopSize;
        private readonly float[] _fftRing;
        private int _fftWrite;
        private int _sinceFft;

        private readonly Complex[] _fft;
        private readonly float[] _window;
        private readonly int _fftM;

        private int[] _binIndex = Array.Empty<int>();
        private float[] _freqWeight = Array.Empty<float>();
        private readonly float[] _barsBack;

        private long _framesProcessed;

        private float _emaEnergy;
        private float _emaVar;
        private double _lastBeatTimeSec;
        private readonly Queue<double> _beatTimes = new();
        private readonly List<double> _intervals = new(16);
        private float _bpmSmoothed = 120f;

        public StreamSpectrumAnalyzer(int barsCount, int fftSize, float minFreq, float maxFreq, int sampleRate)
        {
            _barsCount = barsCount;
            _fftSize = fftSize;
            _minFreq = minFreq;
            _maxFreq = maxFreq;
            _sampleRate = sampleRate;

            _barsBack = new float[_barsCount];

            _fftRing = new float[_fftSize];
            _fft = new Complex[_fftSize];

            _fftM = (int)Math.Round(Math.Log(_fftSize, 2));
            if ((1 << _fftM) != _fftSize)
                throw new ArgumentException("fftSize must be a power of two.");

            _hopSize = Math.Max(64, _fftSize / 8);

            _window = new float[_fftSize];
            for (int i = 0; i < _fftSize; i++)
                _window[i] = 0.5f * (1f - MathF.Cos(MathHelper.TwoPi * i / (_fftSize - 1)));

            BuildLogBins(_sampleRate);
        }

        public void PushMonoSample(float mono)
        {
            _fftRing[_fftWrite] = mono;
            if (++_fftWrite >= _fftSize) _fftWrite = 0;

            _framesProcessed++;
            _sinceFft++;

            if (_sinceFft >= _hopSize)
            {
                _sinceFft = 0;
                ComputeSpectrumAndBeat();
            }
        }

        private void ComputeSpectrumAndBeat()
        {
            int start = _fftWrite;
            for (int i = 0; i < _fftSize; i++)
            {
                float x = _fftRing[start] * _window[i];
                _fft[i].X = x;
                _fft[i].Y = 0f;

                if (++start >= _fftSize) start = 0;
            }

            FastFourierTransform.FFT(true, _fftM, _fft);

            int nyq = _fftSize / 2;

            float domW = 0f;
            float domT = 0f;

            for (int b = 0; b < _barsCount; b++)
            {
                int k = _binIndex[b];

                float mag = 0f;
                for (int kk = k - 1; kk <= k + 1; kk++)
                {
                    float re = _fft[kk].X;
                    float im = _fft[kk].Y;
                    mag += MathF.Sqrt(re * re + im * im);
                }
                mag /= 3f;

                mag *= _freqWeight[b];

                float db = 20f * MathF.Log10(mag + 1e-8f);
                float norm = (db + 60f) / 60f;
                norm = MathHelper.Clamp(norm, 0f, 1f);

                _barsBack[b] = norm;

                float t = b / (float)(_barsCount - 1);
                domW += norm;
                domT += norm * t;
            }

            long samplePos = _framesProcessed;

            float dom = domT / (domW + 1e-6f);
            Vector4 domColor = SpectrumGradient(dom);
            domColor.W = 1f;

            float lowEnergy = LowBandEnergy(30f, 160f, nyq);

            float dt = _hopSize / (float)_sampleRate;
            float aE = 1f - MathF.Exp(-dt / 0.35f);
            float aV = 1f - MathF.Exp(-dt / 0.55f);

            float diff = lowEnergy - _emaEnergy;
            _emaEnergy += diff * aE;
            _emaVar += ((diff * diff) - _emaVar) * aV;

            float sigma = MathF.Sqrt(MathF.Max(_emaVar, 1e-9f));
            float thr = _emaEnergy + 2.2f * sigma;

            double now = _framesProcessed / (double)_sampleRate;
            double minInterval = 0.18;

            if (lowEnergy > thr && (now - _lastBeatTimeSec) > minInterval)
            {
                _lastBeatTimeSec = now;

                float strength = (lowEnergy - thr) / (thr + 1e-6f);
                strength = MathHelper.Clamp(strength, 0f, 1f);

                var beat = new BeatEvent(strength, domColor);
                OnBeatFrame?.Invoke(samplePos, beat);

                UpdateBpm(now);
            }

            OnBarsFrame?.Invoke(samplePos, _barsBack, _bpmSmoothed);
        }

        private float LowBandEnergy(float f0, float f1, int nyq)
        {
            int b0 = (int)(f0 * _fftSize / _sampleRate);
            int b1 = (int)(f1 * _fftSize / _sampleRate);
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
            while (_beatTimes.Count > 12) _beatTimes.Dequeue();
            if (_beatTimes.Count < 4) return;

            _intervals.Clear();

            double prev = double.NaN;
            foreach (var t in _beatTimes)
            {
                if (!double.IsNaN(prev))
                {
                    double d = t - prev;
                    if (d > 0.18 && d < 1.20) _intervals.Add(d);
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

            for (int i = 0; i < _barsCount; i++)
            {
                float t = i / (float)(_barsCount - 1);
                float f = _minFreq * MathF.Pow(ratio, t);

                int k = (int)MathF.Round(f * _fftSize / sampleRate);
                k = Math.Clamp(k, 1, nyq - 2);
                _binIndex[i] = k;
            }

            for (int i = 1; i < _barsCount; i++)
                if (_binIndex[i] <= _binIndex[i - 1])
                    _binIndex[i] = _binIndex[i - 1] + 1;

            if (_binIndex[^1] > nyq - 2)
            {
                _binIndex[^1] = nyq - 2;
                for (int i = _barsCount - 2; i >= 0; i--)
                    if (_binIndex[i] >= _binIndex[i + 1])
                        _binIndex[i] = _binIndex[i + 1] - 1;

                for (int i = 0; i < _barsCount; i++)
                    _binIndex[i] = Math.Max(_binIndex[i], 1);
            }

            _freqWeight = new float[_barsCount];

            const float exponent = 0.40f;
            for (int i = 0; i < _barsCount; i++)
            {
                float freq = _binIndex[i] * sampleRate / (float)_fftSize;
                float w = MathF.Pow(freq / 1000f, exponent);
                w = Math.Clamp(w, 0.35f, 2.8f);
                _freqWeight[i] = w;
            }
        }

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
